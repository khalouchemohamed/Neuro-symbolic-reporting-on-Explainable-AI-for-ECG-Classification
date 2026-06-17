# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ecg_xai_pipeline.config import PipelineConfig
from ecg_xai_pipeline.data import DatasetBundle


# =============================================================================
# MODEL ARCHITECTURE & BLOCKS
# =============================================================================
def conv_block(x, filters: int, kernel_size: int, pool: bool, name: str):
    x = tf.keras.layers.Conv1D(
        filters, kernel_size, padding="same", use_bias=False, name=f"{name}_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)
    if pool:
        x = tf.keras.layers.MaxPooling1D(2, name=f"{name}_pool")(x)
    return x


def tcn_residual_block(
    x, filters: int, kernel_size: int, dilation_rate: int, dropout: float, name: str
):
    residual = x
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(x)
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)

    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters, 1, padding="same", use_bias=False, name=f"{name}_resample"
        )(residual)

    x = tf.keras.layers.Add(name=f"{name}_add")([x, residual])
    return tf.keras.layers.ReLU(name=f"{name}_out")(x)


def cbam_attention(x, name: str, reduction_ratio: int = 8):
    """Full CBAM: channel attention (squeeze-excite) + temporal attention."""
    filters = x.shape[-1]
    reduced = max(filters // reduction_ratio, 1)

    # --- Channel Attention ---
    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_ch_avg")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name=f"{name}_ch_max")(x)

    shared_fc1 = tf.keras.layers.Dense(reduced, activation="relu", name=f"{name}_ch_fc1")
    shared_fc2 = tf.keras.layers.Dense(filters, name=f"{name}_ch_fc2")

    ch_avg = shared_fc2(shared_fc1(avg_pool))
    ch_max = shared_fc2(shared_fc1(max_pool))
    ch_attn = tf.keras.layers.Activation("sigmoid", name=f"{name}_ch_sigmoid")(
        tf.keras.layers.Add(name=f"{name}_ch_add")([ch_avg, ch_max])
    )
    ch_attn = tf.keras.layers.Reshape((1, filters), name=f"{name}_ch_reshape")(ch_attn)
    x = tf.keras.layers.Multiply(name=f"{name}_ch_multiply")([x, ch_attn])

    # --- Temporal (Spatial) Attention ---
    attention = tf.keras.layers.Conv1D(
        1, 1, padding="same", activation="sigmoid", name=f"{name}_weights"
    )(x)
    return tf.keras.layers.Multiply(name=f"{name}_multiply")([x, attention])


def build_hybrid_tcn_cbam_model(
    signal_shape: tuple[int, int], num_classes: int, num_features: int
) -> tf.keras.Model:
    """TCN + temporal attention ECG model with a clinical feature branch.

    The architecture intentionally avoids N x N self-attention and Flatten.
    Gradients remain traceable through local convolutions, dilated context,
    element-wise attention, and global average pooling.
    """
    signal_input = tf.keras.Input(shape=signal_shape, name="ecg_signal")

    x = conv_block(signal_input, 32, 7, pool=True, name="stem1")
    x = conv_block(x, 64, 5, pool=True, name="stem2")
    x = tf.keras.layers.Dropout(0.10, name="stem_dropout")(x)

    for dilation in (1, 2, 4, 8, 16):
        x = tcn_residual_block(
            x,
            filters=128,
            kernel_size=3,
            dilation_rate=dilation,
            dropout=0.10,
            name=f"tcn_d{dilation}",
        )

    x = tf.keras.layers.Conv1D(
        128, 1, padding="same", activation="relu", name="gradcam_target"
    )(x)
    x = cbam_attention(x, name="cbam")
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="signal_norm")(x)
    signal_repr = tf.keras.layers.GlobalAveragePooling1D(name="signal_gap")(x)
    signal_repr = tf.keras.layers.Dense(64, activation="relu", name="signal_dense")(
        signal_repr
    )
    signal_repr = tf.keras.layers.Dropout(0.20, name="signal_dropout")(signal_repr)

    aux_output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="aux_output"
    )(signal_repr)

    feature_input = tf.keras.Input(shape=(num_features,), name="clinical_features")
    feature_repr = tf.keras.layers.Dense(
        16, activation="relu", name="clinical_dense1"
    )(feature_input)
    feature_repr = tf.keras.layers.Dense(
        16, activation="relu", name="clinical_dense2"
    )(feature_repr)

    fused = tf.keras.layers.Concatenate(name="fusion_concat")(
        [signal_repr, feature_repr]
    )
    fused = tf.keras.layers.Dense(64, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(0.30, name="fusion_dropout")(fused)
    main_output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="main_output"
    )(fused)

    return tf.keras.Model(
        inputs=[signal_input, feature_input],
        outputs=[main_output, aux_output],
        name="ecg_tcn_cbam_hybrid",
    )


# =============================================================================
# TRAINING & COMPILATION UTILITIES
# =============================================================================
def set_global_seed(seed: int) -> None:
    """Seed both legacy and modern numpy RNGs plus TensorFlow.

    ``np.random.seed`` covers libraries that still call the legacy
    ``np.random`` namespace internally (e.g. some sklearn routines).
    ``tf.keras.utils.set_random_seed`` seeds Python, NumPy, and TF in one
    call, but does **not** affect ``np.random.default_rng`` instances
    created elsewhere — those are seeded at their own construction site.
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def class_sample_weights(y: np.ndarray) -> tuple[dict[int, float], np.ndarray]:
    classes = np.unique(y)
    raw = compute_class_weight("balanced", classes=classes, y=y)
    capped = np.clip(raw, 1.0, 10.0)
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, capped)}
    weights = np.asarray([class_weights[int(label)] for label in y], dtype=np.float32)
    return class_weights, weights


def sample_weights_from_class_weights(
    y: np.ndarray, class_weights: dict[int, float]
) -> np.ndarray:
    return np.asarray(
        [class_weights.get(int(label), 1.0) for label in y], dtype=np.float32
    )


def compile_model(model: tf.keras.Model, config: PipelineConfig) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss={
            "main_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            "aux_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"main_output": 1.0, "aux_output": 0.30},
        metrics={
            "main_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "aux_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        },
    )


class RRRModel(tf.keras.Model):
    def __init__(self, base_model, rrr_lambda, warmup_epochs, use_aux_head):
        super().__init__()
        self.base_model = base_model
        self.rrr_lambda = rrr_lambda
        self.warmup_epochs = warmup_epochs
        self.use_aux_head = use_aux_head
        self.current_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)

    @property
    def output_names(self):
        return ["main_output", "aux_output"]

    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            return self.base_model([inputs[0], inputs[1]], training=training)
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        if len(data) == 3:
            inputs, targets, sample_weight = data
        else:
            inputs, targets = data
            sample_weight = None

        X, F, mask = inputs[0], inputs[1], inputs[2]
        y_true = targets[0]

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(X)
                main_out, aux_out = self.base_model([X, F], training=True)
                predictions = aux_out if self.use_aux_head else main_out
                
                log_probs = tf.math.log(tf.clip_by_value(predictions, 1e-7, 1.0))
                y_true_flat = tf.reshape(y_true, [-1])
                class_log_probs = tf.reduce_sum(log_probs * tf.one_hot(y_true_flat, depth=5), axis=-1)

            grads_wrt_X = inner_tape.gradient(class_log_probs, X)
            if grads_wrt_X is None:
                grads_wrt_X = tf.zeros_like(X)

            mask_float = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
            rrr_loss = tf.reduce_sum(tf.square(grads_wrt_X * (1.0 - mask_float)), axis=[1, 2])
            mean_rrr_loss = tf.reduce_mean(rrr_loss)

            epoch_float = tf.cast(self.current_epoch, tf.float32)
            warmup_epochs_float = float(self.warmup_epochs)
            if warmup_epochs_float > 0:
                active_lambda = self.rrr_lambda * tf.minimum(1.0, epoch_float / warmup_epochs_float)
            else:
                active_lambda = self.rrr_lambda

            y_pred = [main_out, aux_out]
            loss_value = self.compute_loss(
                x=inputs,
                y=targets,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True
            )
            total_loss = loss_value + active_lambda * mean_rrr_loss

        trainable_vars = self.base_model.trainable_variables
        gradients = outer_tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(targets, y_pred, sample_weight=sample_weight)
        
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = total_loss
        results["rrr_loss"] = mean_rrr_loss
        results["active_lambda"] = active_lambda
        return results

    def test_step(self, data):
        if len(data) == 3:
            inputs, targets, sample_weight = data
        else:
            inputs, targets = data
            sample_weight = None

        X, F, mask = inputs[0], inputs[1], inputs[2]
        main_out, aux_out = self.base_model([X, F], training=False)
        y_pred = [main_out, aux_out]
        
        loss_value = self.compute_loss(
            x=inputs,
            y=targets,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=False
        )
        
        with tf.GradientTape() as tape:
            tape.watch(X)
            main_out_v, aux_out_v = self.base_model([X, F], training=False)
            predictions_v = aux_out_v if self.use_aux_head else main_out_v
            log_probs = tf.math.log(tf.clip_by_value(predictions_v, 1e-7, 1.0))
            y_true_flat = tf.reshape(targets[0], [-1])
            class_log_probs = tf.reduce_sum(log_probs * tf.one_hot(y_true_flat, depth=5), axis=-1)
            
        grads_wrt_X = tape.gradient(class_log_probs, X)
        if grads_wrt_X is None:
            grads_wrt_X = tf.zeros_like(X)
        
        mask_float = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        rrr_loss = tf.reduce_sum(tf.square(grads_wrt_X * (1.0 - mask_float)), axis=[1, 2])
        mean_rrr_loss = tf.reduce_mean(rrr_loss)

        epoch_float = tf.cast(self.current_epoch, tf.float32)
        warmup_epochs_float = float(self.warmup_epochs)
        if warmup_epochs_float > 0:
            active_lambda = self.rrr_lambda * tf.minimum(1.0, epoch_float / warmup_epochs_float)
        else:
            active_lambda = self.rrr_lambda

        total_loss = loss_value + active_lambda * mean_rrr_loss

        self.compiled_metrics.update_state(targets, y_pred, sample_weight=sample_weight)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = total_loss
        results["rrr_loss"] = mean_rrr_loss
        results["active_lambda"] = active_lambda
        return results


class EpochTrackerCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(epoch)


class FunctionalModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_main_output_loss", save_best_only=True, mode="min", verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.is_min = (mode == "min")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        
        improved = (val < self.best_value) if self.is_min else (val > self.best_value)
        if not self.save_best_only or improved:
            self.best_value = val
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1:05d}: {self.monitor} improved from {self.best_value if not self.save_best_only else (self.best_value if not improved else 'inf')} to {val:.5f}, saving base model to {self.filepath}")
            if hasattr(self.model, "base_model"):
                self.model.base_model.save(self.filepath)
            else:
                self.model.save(self.filepath)


def train_or_load_model(
    bundle: DatasetBundle, config: PipelineConfig
) -> tuple[tf.keras.Model, dict[str, list[float]] | None]:
    resume = config.mode == "resume_post_xai" and config.model_path.exists()
    if resume:
        return tf.keras.models.load_model(config.model_path, compile=False), None

    model = build_hybrid_tcn_cbam_model(
        signal_shape=(config.cropped_len, 1),
        num_classes=config.num_classes,
        num_features=config.num_features,
    )
    
    if config.rrr_lambda > 0:
        rrr_model = RRRModel(
            base_model=model,
            rrr_lambda=config.rrr_lambda,
            warmup_epochs=config.rrr_warmup_epochs,
            use_aux_head=config.rrr_use_aux_head
        )
        compile_model(rrr_model, config)
        fit_model = rrr_model
    else:
        compile_model(model, config)
        fit_model = model

    class_weights, train_weights = class_sample_weights(bundle.y_train)
    val_weights = sample_weights_from_class_weights(bundle.y_val, class_weights)
    print(f"Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_main_output_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_main_output_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
    ]

    if config.rrr_lambda > 0:
        callbacks.append(EpochTrackerCallback())
        callbacks.append(FunctionalModelCheckpoint(
            filepath=config.model_path,
            monitor="val_main_output_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ))
    else:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            config.model_path,
            monitor="val_main_output_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ))

    if config.rrr_lambda > 0:
        x_train_fit = [bundle.X_train, bundle.F_train_model, bundle.clinical_masks_train]
        x_val_fit = [bundle.X_val, bundle.F_val, bundle.clinical_masks_val]
    else:
        x_train_fit = [bundle.X_train, bundle.F_train_model]
        x_val_fit = [bundle.X_val, bundle.F_val]

    history = fit_model.fit(
        x_train_fit,
        [bundle.y_train, bundle.y_train],
        validation_data=(
            x_val_fit,
            [bundle.y_val, bundle.y_val],
            [val_weights, val_weights],
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        sample_weight=[train_weights, train_weights],
        callbacks=callbacks,
        verbose=1,
    )
    return tf.keras.models.load_model(config.model_path, compile=False), history.history


# =============================================================================
# PREDICTION UTILITIES
# =============================================================================
def predict_main(model: tf.keras.Model, X: np.ndarray, F: np.ndarray, batch_size: int):
    predictions = model.predict([X, F], batch_size=batch_size, verbose=0)
    main_probs = predictions[0] if isinstance(predictions, list) else predictions
    return main_probs, np.argmax(main_probs, axis=1)


def select_stratified_samples(
    y: np.ndarray, config: PipelineConfig, rng: np.random.Generator
) -> list[int]:
    indices: list[int] = []
    for cls in range(config.num_classes):
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) == 0:
            continue
        n = min(config.samples_per_class, len(cls_indices))
        indices.extend(rng.choice(cls_indices, size=n, replace=False).tolist())
    return sorted(int(idx) for idx in indices)


# =============================================================================
# EXPLAINABLE AI (XAI) ATTRIBUTIONS
# =============================================================================
def main_output_model(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(inputs=model.inputs, outputs=model.outputs[0])


def explanation_output_model(model: tf.keras.Model, use_aux_head: bool) -> tf.keras.Model:
    if use_aux_head and len(model.outputs) > 1:
        return tf.keras.Model(inputs=model.inputs, outputs=model.outputs[0])
    return tf.keras.Model(inputs=model.inputs, outputs=model.outputs[0])


def _selected_class_score(
    predictions: tf.Tensor, predicted_classes: np.ndarray | tf.Tensor, score_mode: str
) -> tf.Tensor:
    pred_tensor = tf.convert_to_tensor(predicted_classes, dtype=tf.int32)
    class_indices = tf.stack([tf.range(tf.shape(predictions)[0]), pred_tensor])
    selected = tf.gather_nd(predictions, tf.transpose(class_indices))

    if score_mode == "probability":
        return selected
    if score_mode == "log_probability":
        return tf.math.log(tf.clip_by_value(selected, 1e-7, 1.0))
    raise ValueError(f"Unsupported XAI score mode: {score_mode}")


def _select_predicted_class(values: np.ndarray, predicted_classes: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    n = len(predicted_classes)
    if values.ndim == 4:
        return values[np.arange(n), :, 0, predicted_classes]
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[:, :, 0]
    if values.ndim == 3:
        return values[np.arange(n), :, predicted_classes]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unsupported attribution shape: {values.shape}")


def _extract_signal_shap(shap_values, predicted_classes: np.ndarray) -> np.ndarray:
    n = len(predicted_classes)
    if isinstance(shap_values, list):
        if shap_values and isinstance(shap_values[0], np.ndarray):
            first = np.asarray(shap_values[0])
            if first.shape[0] == n:
                return _select_predicted_class(first, predicted_classes)

        if shap_values and isinstance(shap_values[0], (list, tuple)):
            selected = []
            for row, cls in enumerate(predicted_classes):
                selected.append(np.asarray(shap_values[int(cls)][0])[row].reshape(-1))
            return np.asarray(selected, dtype=np.float32)

        selected = []
        for row, cls in enumerate(predicted_classes):
            selected.append(np.asarray(shap_values[int(cls)])[row].reshape(-1))
        return np.asarray(selected, dtype=np.float32)

    return _select_predicted_class(np.asarray(shap_values), predicted_classes)


def shap_attributions(
    model: tf.keras.Model,
    X_background: np.ndarray,
    F_background: np.ndarray,
    X_samples: np.ndarray,
    F_samples: np.ndarray,
    predicted_classes: np.ndarray,
    use_aux_head: bool,
):
    import shap

    exp_model = explanation_output_model(model, use_aux_head)
    
    if len(exp_model.inputs) == 1:
        explainer = shap.GradientExplainer(exp_model, X_background)
        shap_values = explainer.shap_values(X_samples)
    else:
        explainer = shap.GradientExplainer(exp_model, [X_background, F_background])
        shap_values = explainer.shap_values([X_samples, F_samples])
        
    return _extract_signal_shap(shap_values, predicted_classes)


def gradcam_attributions(
    model: tf.keras.Model,
    X_samples: np.ndarray,
    F_samples: np.ndarray,
    predicted_classes: np.ndarray,
    layer_name: str = "cbam_multiply",
    use_aux_head: bool = True,
    score_mode: str = "log_probability",
) -> np.ndarray:
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        target_layer = model.get_layer("gradcam_target")

    output_tensor = model.outputs[0]
    grad_model = tf.keras.Model(
        inputs=model.inputs, outputs=[target_layer.output, output_tensor]
    )

    X_tensor = tf.convert_to_tensor(X_samples, dtype=tf.float32)
    F_tensor = tf.convert_to_tensor(F_samples, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        conv_outputs, predictions = grad_model([X_tensor, F_tensor], training=False)
        selected = _selected_class_score(predictions, predicted_classes, score_mode)

    grads, input_grads = tape.gradient(selected, [conv_outputs, X_tensor])
    if grads is None:
        return np.zeros((X_samples.shape[0], X_samples.shape[1]), dtype=np.float32)

    weights = tf.reduce_mean(grads, axis=1, keepdims=True)
    cams = tf.reduce_sum(weights * conv_outputs, axis=-1).numpy()
    cams = np.maximum(cams, 0.0)
    input_gate = None
    if input_grads is not None:
        input_gate = np.maximum((input_grads.numpy() * X_samples)[:, :, 0], 0.0)

    resized = []
    target_len = X_samples.shape[1]
    for idx, cam in enumerate(cams):
        if np.allclose(cam, 0):
            resized.append(np.zeros(target_len, dtype=np.float32))
            continue
        xp = np.linspace(0, target_len - 1, num=len(cam))
        interp = np.interp(np.arange(target_len), xp, cam)
        interp = interp / (interp.max() + 1e-9)

        if input_gate is not None and not np.allclose(input_gate[idx], 0):
            gate = input_gate[idx] / (input_gate[idx].max() + 1e-9)
            interp = interp * (0.25 + 0.75 * gate)
            interp = interp / (interp.max() + 1e-9)

        resized.append(interp.astype(np.float32))
    return np.asarray(resized, dtype=np.float32)


def integrated_gradients_attributions(
    model: tf.keras.Model,
    X_samples: np.ndarray,
    F_samples: np.ndarray,
    predicted_classes: np.ndarray,
    config: PipelineConfig,
    X_background: np.ndarray | None = None,
) -> np.ndarray:
    if X_background is not None:
        baseline = np.tile(np.median(X_background, axis=0), (X_samples.shape[0], 1, 1)).astype(np.float32)
    else:
        baseline = np.zeros_like(X_samples, dtype=np.float32)
    alphas = np.linspace(0.0, 1.0, config.integrated_gradients_steps, dtype=np.float32)
    total_attrs = np.zeros_like(X_samples, dtype=np.float32)

    F_tensor = tf.convert_to_tensor(F_samples, dtype=tf.float32)
    score_model = explanation_output_model(model, config.integrated_gradients_use_aux_head)
    smooth_samples = max(1, int(config.integrated_gradients_smooth_samples))
    rng = np.random.default_rng(config.random_seed + 991)

    for smooth_idx in range(smooth_samples):
        if smooth_idx == 0 or config.integrated_gradients_noise_std <= 0:
            X_target = X_samples.astype(np.float32, copy=True)
        else:
            noise = rng.normal(
                loc=0.0,
                scale=config.integrated_gradients_noise_std,
                size=X_samples.shape,
            ).astype(np.float32)
            X_target = np.clip(X_samples + noise, 0.0, 1.0)

        total_grads = np.zeros_like(X_samples, dtype=np.float32)
        for alpha in alphas:
            interpolated = baseline + alpha * (X_target - baseline)
            X_tensor = tf.constant(interpolated, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                if len(score_model.inputs) == 1:
                    predictions = score_model(X_tensor, training=False)
                else:
                    predictions = score_model([X_tensor, F_tensor], training=False)
                selected = _selected_class_score(
                    predictions, predicted_classes, config.xai_score_mode
                )
            grads = tape.gradient(selected, X_tensor)
            if grads is not None:
                total_grads += grads.numpy()

        total_attrs += (X_target - baseline) * (total_grads / float(len(alphas)))

    return (total_attrs / float(smooth_samples))[:, :, 0].astype(np.float32)


def compute_all_attributions(
    model: tf.keras.Model,
    X_background: np.ndarray,
    F_background: np.ndarray,
    X_samples: np.ndarray,
    F_samples: np.ndarray,
    predicted_classes: np.ndarray,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    return {
        "SHAP": shap_attributions(
            model,
            X_background,
            F_background,
            X_samples,
            F_samples,
            predicted_classes,
            use_aux_head=config.shap_use_aux_head,
        ),
        "GradCAM": gradcam_attributions(
            model,
            X_samples,
            F_samples,
            predicted_classes,
            layer_name=config.gradcam_layer_name,
            use_aux_head=config.gradcam_use_aux_head,
            score_mode=config.xai_score_mode,
        ),
        "IntegratedGradients": integrated_gradients_attributions(
            model, X_samples, F_samples, predicted_classes, config, X_background=X_background
        ),
    }
