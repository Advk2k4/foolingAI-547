# Final Project Report: Adversarial Attacks on EEG-BCI Classifiers

**ECE 547 — Spring 2026**

**Authors:** Aadvik Mishra, Elizabeth Peter, Gautam Jayakishan, Havi Patel

---

## Abstract

Brain-Computer Interface (BCI) systems are becoming increasingly important in healthcare, assistive technologies, neuroprosthetics, and human-computer interaction. Electroencephalography (EEG)-based BCI systems rely on machine learning classifiers to interpret neural activity and translate it into device commands. Recent advances in adversarial machine learning have shown that classifiers can be vulnerable to small, carefully designed perturbations that are imperceptible to human observers, raising serious concerns about the reliability and security of EEG-BCI systems.

This project investigates the vulnerability of EEG-based BCI classifiers to adversarial attacks by analyzing how minimal modifications to raw EEG signals affect classification outcomes. Specifically, we evaluate whether changing only one or two EEG sample points within a signal can significantly alter classification accuracy while remaining imperceptible to users. The study uses an FBCSP (Filter Bank Common Spatial Pattern) feature extraction pipeline combined with an RBF-SVM classifier. The dataset consists of 300 EEG trials with 22 channels sampled at 250 Hz, derived from the publicly available High-Gamma Dataset (Schirrmeister et al., 2017).

Both random and targeted perturbation attacks were implemented at magnitudes of 5%, 10%, 15%, and 20% of each trial's peak-to-peak signal range. Experimental results demonstrate that random attacks produce consistently low misclassification rates (0–2%) across all conditions, while targeted gradient-based attacks substantially degrade classifier performance. Two-point targeted perturbations achieved the highest impact, raising misclassification rates to 21.6% at 20% perturbation magnitude. Critically, all attack conditions maintained Signal-to-Noise Ratio (SNR) values above the 20 dB imperceptibility threshold, confirming that the perturbations remain undetectable under standard signal quality criteria.

These findings demonstrate that EEG-BCI systems are significantly more vulnerable to structured adversarial attacks than to random noise, and highlight important security concerns for real-world BCI deployment.

---

## 1. Introduction

Brain-Computer Interfaces (BCIs) represent a rapidly growing class of technology that creates a direct communication channel between the human brain and external devices. Unlike traditional human-computer interaction which relies on muscular movement, BCIs decode neural signals directly, enabling control of prosthetic limbs, wheelchairs, communication devices, and computer systems by thought alone. Among the various neural recording modalities, Electroencephalography (EEG) has emerged as the dominant approach for non-invasive BCIs due to its relatively low cost, portability, high temporal resolution, and absence of surgical risk.

EEG-based BCIs have found particularly important applications in the medical field. For patients with conditions such as amyotrophic lateral sclerosis (ALS), spinal cord injury, or locked-in syndrome, a reliable BCI may represent the primary or only means of communication with the outside world. Prosthetic limb control systems guided by EEG signals are increasingly being tested in clinical trials, and EEG-guided neurofeedback has shown promise in stroke rehabilitation. As these technologies move from research laboratories into clinical and home settings, the reliability and security of the underlying classification systems becomes critically important.

The classification performance of EEG-BCI systems depends entirely on machine learning models trained to recognize patterns in neural signals. Motor imagery BCIs, the focus of this project, work by detecting characteristic changes in EEG power spectra when users imagine performing physical movements, such as moving their left hand, right hand, or feet. These imagined movements produce spatially and spectrally distinct patterns that classifiers can learn to distinguish. However, EEG signals are inherently noisy, highly individual, and sensitive to changes in mental state, electrode placement, and environmental conditions — all of which make classification a challenging problem.

Recent developments in adversarial machine learning have demonstrated that classification models, even highly accurate ones, can be reliably fooled by carefully designed input perturbations that are imperceptible to human observers. While this phenomenon has been extensively studied in the domains of image recognition and natural language processing, its implications for EEG-based systems remain relatively underexplored. The question is both practically and ethically significant: if an adversary could reliably cause a prosthetic limb to move in the wrong direction, or prevent a locked-in patient from communicating, the consequences could be severe.

This project directly addresses this question. We investigate whether minimal adversarial modifications — specifically, changing only one or two sample points within a trial containing 1001 time points — are sufficient to cause the FBCSP+SVM classifier to misclassify the signal. We compare two attack strategies: a random attack that serves as a baseline, and a gradient-based targeted attack that exploits knowledge of the classifier's internal structure. Both attack types are evaluated under strict imperceptibility constraints using SNR analysis, and the results are analyzed to understand which trials and which classes are most vulnerable.

The primary contributions of this work are:

1. Demonstration that targeted gradient-based attacks on FBCSP+SVM EEG classifiers can achieve up to 21.6% misclassification by modifying only two sample points out of 1001.
2. A detailed vulnerability analysis identifying which trials and classes are most susceptible and explaining why in terms of the classifier's feature space geometry.
3. A systematic comparison of random versus targeted attacks across four perturbation magnitudes and two attack point counts, showing targeted attacks are up to 10.8× more effective.
4. Confirmation that all tested perturbations remain imperceptible (SNR > 20 dB), establishing that the vulnerability is practically exploitable without detection under current signal quality standards.

---

## 2. Background

### 2.1 Electroencephalography and Motor Imagery

Electroencephalography measures the electrical potential differences generated by synchronized neural activity, recorded from electrodes placed on the scalp. EEG signals are typically analyzed in distinct frequency bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30+ Hz). Motor imagery tasks produce characteristic changes in the alpha and beta frequency bands, specifically event-related desynchronization (ERD) in the contralateral motor cortex and event-related synchronization (ERS) in the ipsilateral cortex. These spatial and spectral patterns form the basis for motor imagery classification.

EEG signals present numerous challenges for classification: they are non-stationary, have low signal-to-noise ratio, exhibit significant variability across subjects and sessions, and contain artifacts from muscle activity, eye movements, and environmental electrical noise. These characteristics make EEG classification particularly challenging and motivate the use of sophisticated feature extraction methods.

### 2.2 Filter Bank Common Spatial Pattern (FBCSP)

The Filter Bank Common Spatial Pattern (FBCSP) algorithm is one of the most widely used approaches for motor imagery EEG classification. It extends the basic Common Spatial Pattern (CSP) method by applying CSP independently across multiple frequency bands, capturing discriminative information across the full motor imagery spectrum.

The CSP algorithm computes spatial filters W such that the log-variance of the spatially filtered signal is maximally discriminative between classes. Formally, given EEG trials from two classes, CSP solves a generalized eigenvalue problem:

**Σ₁ w = λ Σ₂ w**

where Σ₁ and Σ₂ are the class-wise covariance matrices. The resulting filters maximize variance for one class while minimizing it for the other, yielding features that capture spatially distinct neural patterns.

FBCSP applies this procedure independently for each frequency band in a filter bank. In our implementation, six frequency bands are used: 4–8 Hz, 8–12 Hz, 12–16 Hz, 16–20 Hz, 20–24 Hz, and 24–30 Hz. For each band, six CSP components are extracted, yielding 36 total features. These features are log-variance transformed to stabilize variance and normalize the distribution before being passed to the classifier.

### 2.3 Support Vector Machines

Support Vector Machines (SVMs) are supervised learning models that find the optimal separating hyperplane between classes in a high-dimensional feature space. For linearly non-separable problems, the kernel trick maps inputs to a higher-dimensional space where linear separation becomes possible. The Radial Basis Function (RBF) kernel computes similarity between points as:

K(x, x') = exp(−γ ||x − x'||²)

where γ controls the influence radius of each training point. The SVM optimization problem seeks to maximize the margin between the decision boundary and the nearest training points (support vectors), subject to a regularization parameter C that controls the trade-off between margin width and training error.

For multi-class problems, scikit-learn's SVC uses a one-versus-one (OvO) strategy, training a binary classifier for each pair of classes and using majority voting to make final predictions.

### 2.4 Adversarial Attacks in Machine Learning

Adversarial attacks are input perturbations designed to cause misclassification in machine learning models. Goodfellow et al. (2014) introduced the Fast Gradient Sign Method (FGSM), which computes the gradient of the loss with respect to the input and perturbs the input in the direction that maximally increases the loss:

x_adv = x + ε · sign(∇ₓ J(θ, x, y))

Projected Gradient Descent (PGD), introduced by Madry et al. (2018), extends FGSM with multiple iterative steps and projection back onto a perturbation budget. The Carlini-Wagner (C&W) attack frames adversarial example generation as an optimization problem that minimizes perturbation magnitude while achieving misclassification.

While these methods were developed primarily for image classification, the underlying principles apply to any differentiable or approximately differentiable classifier. Our targeted attack is conceptually related to FGSM but adapted for the FBCSP+SVM pipeline, which is not directly differentiable and requires a hybrid numerical-analytic gradient computation approach.

### 2.5 Imperceptibility and SNR

Signal-to-Noise Ratio (SNR) is a widely used metric for quantifying signal quality. In this project we use the power-based SNR formulation:

SNR (dB) = 10 · log₁₀(P_signal / P_noise)

where P_signal = mean(x²) and P_noise = mean((x_perturbed − x)²). A threshold of 20 dB is used as the imperceptibility criterion, borrowed from psychoacoustics. We acknowledge that this threshold was originally derived for audio signals and may not perfectly characterize EEG imperceptibility; however, it serves as a conservative lower bound. Typical clinical EEG recordings operate at SNR of 15–25 dB due to electrode artifacts and muscle activity, so our perturbations — which achieve SNR of 36–48 dB across all conditions — are substantially smaller than naturally occurring clinical noise.

---

## 3. Problem Statement

EEG-based Brain-Computer Interface classifiers may be highly sensitive to small adversarial modifications of raw EEG signals, posing security and reliability risks in real-world applications. The key questions addressed in this project are:

1. Can adversarial perturbations limited to one or two sample points in a 1001-point trial cause meaningful misclassification in an FBCSP+SVM classifier?
2. Does targeted gradient-based perturbation significantly outperform random perturbation of equal magnitude?
3. Do these perturbations remain imperceptible under SNR-based imperceptibility criteria?
4. Which trials and classes are most vulnerable, and why?

---

## 4. Project Objectives

This project is primarily concerned with exploring the vulnerability of EEG-BCI classifiers to adversarial attacks. Specifically, the aims are to:

- Examine the influence of random and targeted attacks on EEG classification accuracy
- Explore the effectiveness of single-point versus two-point attacks
- Investigate the relationship between perturbation magnitude and misclassification rate
- Demonstrate that all attacks remain imperceptible using SNR analysis
- Assess the robustness of an FBCSP+RBF-SVM classification pipeline
- Provide a vulnerability analysis identifying which trials and classes are most susceptible

---

## 5. Dataset

### 5.1 Source and Provenance

The EEG data used in this project is derived from the publicly available High-Gamma Dataset, introduced by Schirrmeister et al. in their 2017 paper *Deep learning with convolutional neural networks for EEG decoding and visualization* (Human Brain Mapping, 38(11), 5391–5420). The dataset is hosted on the G-Node GIN repository and contains high-density 128-channel EEG recordings at 250 Hz from 14 subjects performing four-class motor imagery tasks: left hand, right hand, feet, and rest.

### 5.2 Trial Selection and Preprocessing

For this project, 300 trials were sequentially extracted from Subjects 1 and 2 of the dataset. The trial selection was not randomized across all 14 subjects; rather, trials were drawn starting from Subject 1 and continuing into Subject 2 until 300 were obtained. This means the dataset is effectively a near-within-subject collection, concentrated primarily on the neurological signatures of two specific individuals.

This subject pool composition directly explains the high baseline accuracy of 89.5%. Within-subject EEG classification is substantially easier than cross-subject classification because the classifier can learn subject-specific spatial and spectral patterns. Had trials been drawn from all 14 subjects, baseline accuracy would be expected to fall in the 50–65% range due to the large inter-subject variability in EEG patterns. This scope is acknowledged as a limitation; cross-subject generalization is identified as a direction for future work.

The raw 128-channel montage was spatially downsampled to a standard 22-channel 10-20 layout to match the BCI Competition IV 2a benchmark format. This downsampling focuses the channel space on the motor cortex and reduces computational complexity without discarding the most diagnostically relevant electrodes.

Raw epochs were 4-second windows (1000 time points at 250 Hz). A single zero-padding sample was appended to the time dimension, yielding 1001 time points per trial, for structural compatibility with the input pipeline. The final delivered dataset has shape (300, 22, 1001).

EEG signals were originally stored in microvolts and converted to volts (×10⁻⁶) before processing, as required by the MNE library and the FBCSP pipeline operating in SI units. After conversion, signal values spanned approximately −8.23×10⁻⁵ to 7.92×10⁻⁵ V, giving an overall peak-to-peak range of approximately 161 µV across all trials.

### 5.3 Class Distribution and Exclusion

The original dataset contains four motor imagery classes: feet, left hand, right hand, and rest. The right hand class was excluded from this project due to poor separability from left hand, which reduced baseline accuracy to 66.67% when included. Excluding right hand yielded a 3-class problem (feet=0, left_hand=1, rest=2) with 225 usable trials. This exclusion is consistent with prior work that finds right/left hand CSP features less separable in within-subject settings when the dataset is small.

### 5.4 Train/Test Split

A stratified 75/25 split was applied using random seed 42, yielding approximately 169 training trials and 56 test trials. Stratification ensures class balance is maintained in both partitions. No separate validation set was used; hyperparameters were selected based on established motor imagery BCI literature rather than data-driven grid search.

---

## 6. Methodology

### 6.1 System Overview

The experimental pipeline consists of the following stages:

1. Load and preprocess EEG data (µV→V conversion, class filtering)
2. Train FBCSP+SVM classifier on training set
3. Evaluate baseline accuracy on test set
4. Filter to correctly classified test trials
5. Apply adversarial perturbations across all experimental conditions
6. Measure misclassification rate and SNR for each condition
7. Analyze results by attack type, magnitude, point count, and trial

### 6.2 Feature Extraction: FBCSP

The FBCSP implementation uses six frequency bands: 4–8, 8–12, 12–16, 16–20, 20–24, and 24–30 Hz. Each band is extracted using a 4th-order Butterworth bandpass filter applied with zero-phase filtering (forward and reverse pass). For each band, six CSP components are computed using MNE's CSP implementation with Ledoit-Wolf covariance regularization, which provides stable covariance estimates under small sample sizes. Log-variance features are computed from the spatially filtered signal, producing 6 features per band and 36 features total.

Features are normalized using scikit-learn's StandardScaler, which centers and scales each feature to zero mean and unit variance using statistics computed on the training set.

### 6.3 Classifier: RBF-SVM

The classifier is a Support Vector Machine with an RBF kernel, trained using scikit-learn's SVC with parameters C=100 and γ='scale'. The probability=True setting enables Platt scaling, which produces calibrated class probability estimates used in the targeted attack gradient computation. Multi-class classification uses the one-versus-one (OvO) strategy.

Hyperparameters (C=100, RBF kernel, 6 CSP components per band) were selected based on established practice in motor imagery BCI literature and were not tuned via grid search. The model achieved a baseline test accuracy of 89.5% (51/57 correctly classified trials).

### 6.4 Perturbation Design

#### 6.4.1 Perturbation Magnitude

Rather than using fixed absolute magnitudes, perturbation magnitude is defined as a percentage of each individual trial's peak-to-peak amplitude. This proportional approach ensures that perturbations are calibrated to the natural dynamic range of each trial, making comparisons across trials more meaningful.

Individual trial peak-to-peak amplitudes ranged from 30.54 µV to 133.56 µV (mean: 40.52 µV). At a 5% perturbation level, a trial with a 30 µV range receives a 1.5 µV perturbation, while a trial with a 130 µV range receives a 6.5 µV perturbation. The four perturbation magnitudes tested are 5%, 10%, 15%, and 20% of each trial's peak-to-peak range.

#### 6.4.2 Random Attack

The random attack selects n time points uniformly at random from the 1001-point trial and applies a perturbation of ±magnitude_uv at each selected point. The sign (positive or negative) of each perturbation is drawn uniformly at random, independently per point. This attack does not use any information about the classifier and serves as a baseline representing unstructured noise injection.

#### 6.4.3 Targeted Attack

The targeted attack uses gradient information from the FBCSP+SVM pipeline to identify the most impactful time point(s) and perturbation direction. The computation proceeds in five stages:

**Stage 1 — Forward pass:** The trial is passed through the full FBCSP+SVM pipeline to obtain class probability estimates from the SVM's Platt-scaled output. The predicted class (highest probability) and runner-up class (second highest) are identified.

**Stage 2 — Numerical gradient in feature space:** A finite-difference numerical gradient is computed in the 36-dimensional scaled feature space. For each of the 36 features, two SVM probability evaluations are performed with the feature perturbed by ±ε (ε = 10⁻⁴), yielding the gradient of (P_runner_up − P_predicted) with respect to each scaled feature. This requires 72 SVM probability evaluations per trial.

**Stage 3 — Backpropagation through StandardScaler:** The gradient in scaled feature space is converted to gradient in raw feature space by element-wise division by the learned per-feature standard deviations: grad_raw = grad_scaled / σ.

**Stage 4 — Backpropagation through FBCSP:** For each of the six frequency bands, the gradient is analytically propagated back through the CSP transform. Given spatial filters W (shape: n_components × n_channels), band-filtered signal X_band, and spatially filtered signal Z = W @ X_band:

```
v = mean(Z², axis=time)          # per-component variance
α = 2 · w_k / (v · T)            # per-component scaling
grad_trial += W.T @ diag(α) @ Z  # backprop to signal space
```

where w_k is the feature-space gradient for this band's components and T is the number of time points. Gradients from all six bands are summed.

**Stage 5 — Time point selection and perturbation:** The resulting gradient matrix has shape (22 channels × 1001 time points). For each time point, the L2 norm of the gradient vector across all 22 channels is computed. The time point(s) with the largest gradient norm are selected for perturbation. For each selected time point, the perturbation is applied in the gradient direction (normalized to unit length) scaled by the perturbation magnitude:

```
x[:, t*] += magnitude_v · grad[:, t*] / ||grad[:, t*]||
```

This is conceptually analogous to the Fast Gradient Sign Method (FGSM) from image adversarial attack literature, adapted to operate in the EEG time-domain signal space through the composite FBCSP+SVM pipeline.

### 6.5 Experiment Grid

All 51 correctly classified test trials were used as the basis for experiments. The 6 misclassified trials were excluded to ensure that observed misclassifications in experiments are attributable to the perturbation rather than pre-existing classifier errors.

The full experiment grid is:

- **Trials:** 51 (correctly classified)
- **Attack types:** random, targeted (2)
- **Point counts:** 1, 2 (2)
- **Magnitudes:** 5%, 10%, 15%, 20% (4)
- **Total conditions per trial:** 16
- **Total experiments:** 51 × 16 = 816

Results for each experiment are recorded in experiment_log.csv with fields: trial_id, attack_type, n_points, perturbation_pct, magnitude_uv, orig_pred, pert_pred, misclassified, snr_db, trial_range_uv. The full experiment sweep completed in approximately 22 seconds.

### 6.6 Evaluation Metrics

- **Misclassification rate:** Fraction of trials where the perturbed prediction differs from the original prediction, reported per experimental condition.
- **SNR (dB):** Power-based SNR computed as 10·log₁₀(mean(x²) / mean((x_pert − x)²)), measuring perturbation imperceptibility.
- **95% Confidence Intervals:** Wilson score intervals computed on misclassification rates to account for the small test set size (N=51).

---

## 7. Related Work

### 7.1 Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces

The primary reference for this project is Meng et al. (2024), *Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces*. This paper demonstrates that EEG classifiers are vulnerable to adversarial perturbations constructed through adversarial filtering, where the entire EEG signal is passed through a learned adversarial filter that preserves overall signal characteristics while systematically deceiving the classifier. The paper achieves attack success rates of approximately 90%, but does so by modifying every sample point in the signal.

Our work differs in its more constrained attack setting: we modify only one or two sample points out of 1001, representing a 0.1–0.2% modification of the time series. While this yields lower absolute misclassification rates (up to 21.6% vs. 90%), the extreme sparsity of our attack is a key contribution — demonstrating that meaningful vulnerability exists even under very strict perturbation constraints.

### 7.2 EEGNet: A Compact CNN for EEG-Based BCIs

Lawhern et al. (2018), *EEGNet: A Compact CNN for EEG-Based BCIs*, introduces a lightweight convolutional neural network for EEG classification using depthwise and separable convolutions to learn spatial and temporal features from raw EEG. EEGNet achieves competitive performance across multiple BCI paradigms with far fewer parameters than larger deep learning models, making it suitable for embedded systems.

While our project uses FBCSP+RBF-SVM rather than a CNN architecture, EEGNet is relevant for two reasons. First, it demonstrates that modern EEG classifiers are increasingly end-to-end trainable, making them directly susceptible to gradient-based adversarial attacks. Second, Lawhern et al. note that oscillatory BCI features without strong lateralization are more variable and harder to separate — a finding consistent with our observation that feet and rest classes are more confusable than left hand in our FBCSP feature space.

### 7.3 Adversarial Attacks Against Deep Neural Networks Based Brain-Computer Interfaces

Zhang and Wu (2019), *Adversarial Attacks Against Deep Neural Networks Based Brain-Computer Interfaces*, demonstrate that deep neural network EEG classifiers are vulnerable to adversarial examples adapted from computer vision attack methods. The paper explores both white-box and black-box attack settings and shows that adversarial examples can transfer across different classifier architectures.

This work directly motivated our project by establishing that the adversarial attack threat is real for EEG systems. However, the paper focuses exclusively on deep neural network classifiers. Our project extends the threat analysis to traditional machine learning pipelines (FBCSP+SVM), demonstrating that the vulnerability is not limited to neural network architectures.

### 7.4 Retraining and Evaluation of Machine Learning and Deep Learning Models for Seizure Classification from EEG Data

Carvajal-Dossman et al. evaluate and compare machine learning and deep learning classifiers for EEG-based seizure detection, emphasizing the importance of preprocessing, feature extraction, and model selection when working with noisy biomedical signals. Their work highlights that EEG classification performance is highly sensitive to the quality and consistency of the preprocessing pipeline — a finding that reinforces the importance of our FBCSP feature extraction design choices.

Although this paper addresses seizure classification rather than adversarial attacks, it provides important context for understanding the practical limitations of EEG classification systems and the real-world consequences of classification errors in healthcare settings.

---

## 8. Experiments and Results

### 8.1 Baseline Classification

The FBCSP+SVM classifier achieved a baseline test accuracy of 89.5%, correctly classifying 51 of 57 test trials. The 6 misclassified trials were excluded from all subsequent adversarial experiments. The 51 correctly classified trials were distributed as follows: 17 feet, 19 left hand, and 15 rest.

The high baseline accuracy reflects the near-within-subject nature of the dataset (Subjects 1 and 2 only). FBCSP+SVM trained on two subjects' data has effectively memorized the motor cortex signatures of those specific individuals.

### 8.2 Random Attack Results

Random attacks produced consistently low misclassification rates across all perturbation conditions.

| Perturbation | 1-Point | 2-Point |
|---|---|---|
| 5% | 2.0% | 0.0% |
| 10% | 0.0% | 2.0% |
| 15% | 0.0% | 2.0% |
| 20% | 2.0% | 2.0% |

These results confirm that the FBCSP+SVM pipeline is robust against random unstructured noise. This is expected: FBCSP features are computed as log-variance over the entire time series, meaning a perturbation at a single randomly selected time point contributes only minimally to the aggregate feature value.

### 8.3 Targeted Attack Results

Targeted gradient-based attacks produced substantially higher misclassification rates.

**One-Point Targeted Attacks:**

| Perturbation | Flipped | Rate | 95% CI |
|---|---|---|---|
| 5% | 1/51 | 2.0% | [0.4%, 10.4%] |
| 10% | 3/51 | 5.9% | [2.0%, 16.0%] |
| 15% | 6/51 | 11.8% | [5.6%, 23.1%] |
| 20% | 6/51 | 11.8% | [5.6%, 23.1%] |

**Two-Point Targeted Attacks:**

| Perturbation | Flipped | Rate | 95% CI |
|---|---|---|---|
| 5% | 3/51 | 5.9% | [2.0%, 16.0%] |
| 10% | 6/51 | 11.8% | [5.6%, 23.1%] |
| 15% | 9/51 | 17.6% | [9.8%, 29.4%] |
| 20% | 11/51 | 21.6% | [12.5%, 34.0%] |

Note: Confidence intervals are wide due to the small test set size (N=51). This is disclosed as a limitation.

### 8.4 SNR Analysis

All perturbation conditions maintained SNR well above the 20 dB imperceptibility threshold.

| Perturbation | Random SNR | Targeted SNR |
|---|---|---|
| 5% | ~35 dB | ~48 dB |
| 10% | ~29 dB | ~42 dB |
| 15% | ~25 dB | ~39 dB |
| 20% | ~23 dB | ~36 dB |

Two observations are notable. First, targeted attacks consistently achieve higher SNR than random attacks at equal perturbation magnitude. At 20%, targeted 1-point attacks averaged 37.7 dB compared to 24.3 dB for random 1-point attacks. This occurs because targeted attacks concentrate all perturbation energy in a single gradient-aligned direction across all 22 channels, while random attacks distribute energy with random signs, creating more total noise power.

Second, 12 random attack experiments on low-amplitude trials fell below the 20 dB threshold (minimum 12.8 dB). These occurred on trials with small peak-to-peak amplitude (e.g., trial 48 had a range of 30.54 µV), where a percentage-based perturbation constitutes a larger fraction of the total signal energy. None of these below-threshold experiments resulted in misclassification, so they do not affect the core findings, but they are noted.

### 8.5 Vulnerability Analysis by Trial

Of the 51 correctly classified test trials, only 11 (21.6%) were successfully flipped by at least one targeted attack condition. The remaining 40 trials (78.4%) resisted all attack conditions. This indicates that vulnerability is concentrated in a specific subset of trials rather than uniformly distributed.

The three most vulnerable trials were:

- **Trial 38:** Flipped under all 8 targeted conditions (all 4 magnitudes, both 1-point and 2-point). Originally classified as feet (class 0), consistently flipped to rest (class 2).
- **Trial 20:** Flipped under 7 of 8 targeted conditions. Originally feet, consistently flipped to rest.
- **Trial 46:** Flipped under 7 of 8 targeted conditions. Originally feet, consistently flipped to rest.

These trials likely sit near the SVM decision boundary in the 36-dimensional FBCSP feature space, meaning even a small perturbation in the correct direction is sufficient to cross it. The 40 robust trials are presumably located deeper within their class regions, far from any decision surface.

### 8.6 Misclassification Direction Analysis

Examining the 45 successful targeted misclassifications reveals a strong directional pattern:

- **27 (60%):** Feet → Rest
- **16 (35.6%):** Rest → Feet
- **2 (4.4%):** Left Hand → Rest
- **0 (0%):** Any → Left Hand

No attack ever caused a trial to be classified as left hand, and very few left hand trials were flipped at all. This pattern is consistent with the geometry of motor imagery features: left hand motor imagery activates the right motor cortex, producing a strongly lateralized spatial pattern that CSP separates cleanly. Feet and rest imagery produce more broadly distributed or centrally located spatial patterns, occupying neighboring regions in the FBCSP feature space and therefore more easily confused by small perturbations. Lawhern et al. (2018) noted a similar observation: oscillatory EEG features without strong left-right lateralization tend to be more variable and harder to separate.

### 8.7 One-Point Attack Plateau

The one-point targeted attack success rate plateaus at 11.8% between 15% and 20% perturbation magnitude, while the two-point attack continues increasing (17.6% at 15%, 21.6% at 20%). This plateau is a real phenomenon, not measurement noise. The gradient-based selection always identifies the single best time point for each trial. Once that point's perturbation is large enough to flip a trial, further magnitude increases do not flip additional trials — they only affect already-flipped trials more strongly. The second perturbation point in the two-point attack opens attack surface in different time regions or frequency bands that the first point cannot reach, allowing misclassification rates to continue climbing.

This finding suggests a natural ceiling for single-point attacks in this setting (~12% success rate) and implies that attack surface scales with the number of perturbed points.

### 8.8 Comparison with Prior Work

Our results extend prior findings while operating under substantially more constrained conditions. Meng et al. (2024) achieved ~90% attack success rates using full-signal adversarial filtering. Zhang and Wu (2019) demonstrated significant degradation of deep neural network EEG classifiers using image-domain adversarial methods. Our contribution is showing that a perturbation limited to 1–2 sample points out of 1001 (0.1–0.2% of the time series) still achieves meaningful misclassification rates of up to 21.6% in a traditional FBCSP+SVM pipeline. This demonstrates that adversarial vulnerability in EEG systems is not limited to large-scale signal manipulation or deep learning architectures.

---

## 9. Analysis and Reflections

### 9.1 Attack Structure vs. Magnitude

The most significant finding of this project is that attack structure — specifically, whether perturbation is gradient-guided or random — matters far more than perturbation magnitude. At 20% perturbation, the targeted 2-point attack achieves 21.6% misclassification compared to 2% for the random attack of equal magnitude. This 10.8× difference demonstrates that the gradient computation is essential: magnitude alone is insufficient to compromise the classifier.

This has practical security implications. It means an attacker who can compute or approximate the gradient of the classifier's output with respect to the input signal can cause meaningful misclassification with very small absolute perturbations. Conversely, random noise or signal interference of equal magnitude has negligible impact, which is actually reassuring — it means EEG classifiers are inherently robust to everyday environmental noise.

### 9.2 FBCSP Feature Space Geometry

The directional pattern in misclassifications (feet ↔ rest, but not left hand) reveals something important about the FBCSP feature space geometry. The 36 features computed by FBCSP capture spatial power in multiple frequency bands. Left hand motor imagery produces a strongly lateralized pattern (right motor cortex desynchronization) that is spatially distinctive and well-separated from the other classes in feature space. Feet and rest, which involve less distinctly lateralized neural patterns, occupy adjacent regions in feature space and are therefore more easily confused.

This suggests that adversarial vulnerability in EEG classifiers is not uniform across classes — classes with less spatially distinctive neural signatures are inherently more vulnerable to targeted attack. This is a useful insight for BCI system designers: tasks that produce strongly lateralized and distinctive EEG patterns may be more robust to adversarial manipulation.

### 9.3 Imperceptibility and the SNR Threshold

All experimental conditions maintained SNR above 20 dB, confirming that the attacks remain imperceptible under this criterion. The higher SNR of targeted attacks compared to random attacks at equal magnitude (e.g., 37.7 dB vs. 24.3 dB at 20%) is particularly notable. The targeted attack is not just more effective — it is also more efficient, achieving greater misclassification with less signal disruption. This is because the gradient-guided perturbation concentrates energy in a single direction, minimizing the total noise power while maximizing its effect on the classifier output.

We acknowledge that the 20 dB threshold is borrowed from audio psychoacoustics and may not perfectly characterize EEG imperceptibility. A neurologist reviewing raw EEG traces might notice unusual features at 20% perturbation magnitude, even if the SNR criterion is satisfied. Future work should validate imperceptibility through human expert inspection and frequency-domain analysis.

### 9.4 Limitations

Several limitations of this study should be acknowledged:

1. **Small test set:** With N=51 test trials, confidence intervals on misclassification rates are wide (up to ±12 percentage points at 95% confidence). The point estimates reported should be interpreted with this uncertainty in mind.

2. **Two-subject dataset:** Using data from only 2 of 14 available subjects creates a near-within-subject setting. Results may not generalize to cross-subject deployment scenarios, where baseline accuracy and adversarial vulnerability may differ substantially.

3. **White-box attack assumption:** The targeted attack requires full knowledge of the classifier architecture, trained weights, and feature extraction pipeline. Real-world attackers may only have black-box access (ability to query the classifier without internal knowledge), which would require different attack strategies such as transfer attacks or zeroth-order optimization methods.

4. **No human perceptual validation:** Imperceptibility was assessed using SNR only. Human expert inspection and artifact detection analyses were not conducted, which limits our ability to make strong claims about practical undetectability.

5. **No defense evaluation:** This work demonstrates the attack but does not test countermeasures. The practical exploitability of the attack in the presence of defenses remains unknown.

---

## 10. Conclusions

This project investigated the vulnerability of EEG-based Brain-Computer Interface classifiers to minimal adversarial perturbations, specifically modifications limited to one or two sample points in a 1001-point EEG trial. Using an FBCSP+RBF-SVM classification pipeline trained on motor imagery data from the High-Gamma Dataset (Schirrmeister et al., 2017), we conducted 816 experiments across four perturbation magnitudes, two attack types, and two point counts.

The key findings are:

1. **Targeted attacks are substantially more effective than random attacks.** At 20% perturbation magnitude, targeted 2-point attacks achieve 21.6% misclassification compared to 2% for random attacks — a 10.8× improvement — demonstrating that gradient-guided perturbation is essential.

2. **Single-point attacks plateau at ~12% success rate.** The one-point targeted attack plateaus at 11.8% misclassification between 15% and 20% magnitude, while two-point attacks continue scaling to 21.6%. This suggests a natural ceiling for single-point attacks that multi-point attacks can overcome.

3. **Vulnerability is concentrated in specific trials and class pairs.** Only 11 of 51 trials (21.6%) were successfully flipped. Misclassifications were almost entirely confined to the feet-rest class pair, reflecting the proximity of these classes in FBCSP feature space due to similar spatial neural patterns.

4. **All attacks remain imperceptible.** SNR values ranged from 23 to 48 dB across conditions, all above the 20 dB imperceptibility threshold, confirming that the attacks satisfy signal quality criteria.

These findings highlight that EEG-BCI systems have real adversarial vulnerabilities that go beyond susceptibility to random noise, and that gradient-based attacks can exploit classifier structure to achieve disproportionate misclassification with minimal signal modification.

---

## 11. Future Work

Several directions for future work emerge from this study:

**Adversarial defenses.** The most immediate practical follow-up is evaluating mitigation strategies. Adversarial training — augmenting the training set with adversarial examples — is a well-established defense in image classification and could be adapted for EEG. Input anomaly detection, which flags trials where any time point deviates sharply from learned signal statistics, could detect attacks before they reach the classifier. Ensemble methods using majority voting across multiple independent classifiers could reduce per-classifier vulnerability.

**Black-box attacks.** The current attack assumes white-box access to the classifier. Real-world attackers may only be able to query the classifier. Transfer attacks — generating adversarial examples against a substitute model and applying them to the target — and zeroth-order gradient estimation methods are important directions for evaluating more realistic threat scenarios.

**Cross-subject generalization.** Expanding the dataset to all 14 available subjects and evaluating the attack in a cross-subject setting would better reflect real-world BCI deployment, where classifiers typically serve multiple users.

**Alternative classifiers.** Testing the attack against EEGNet (Lawhern et al., 2018) and other deep learning architectures would reveal whether the vulnerability is specific to traditional machine learning pipelines or extends to modern neural network approaches.

**Iterative attacks.** Extending the current single-step attack to iterative methods analogous to PGD, which refine the perturbation over multiple steps, may achieve higher success rates within the same imperceptibility budget.

**Human perceptual validation.** Validating imperceptibility through expert neurologist inspection and automated artifact detection algorithms would strengthen the imperceptibility claims beyond the SNR metric alone.

---

## 12. Partition of Work and Schedule

| Week | Tasks |
|---|---|
| Week of 04/04 | Dataset collection (Gautam); Preprocessing pipeline (Aadvik) |
| Week of 04/12 | Preprocessing (Aadvik); Model training (Aadvik); Dataset preparation (Gautam); Experiment design (Havi) |
| Week of 04/19 | Model training (Aadvik); Dataset preparation (Gautam); Experiment sweep (Havi); Analysis (Elizabeth) |
| Week of 04/26 | Perturbation framework (Havi); Gradient attack implementation (Aadvik); Results compilation |
| Week of 05/03 | Figure generation (Elizabeth); Report writing (all); Final review |

---

## References

1. Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.

2. Meng, L., Jiang, X., Chen, X., Liu, W., Luo, H., & Wu, D. (2024). Adversarial filtering based evasion and backdoor attacks to EEG-based brain-computer interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*.

3. Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., & Lance, B.J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.

4. Zhang, X., & Wu, D. (2019). Adversarial attacks against deep neural networks based brain-computer interfaces. *arXiv preprint arXiv:1912.01875*.

5. Carvajal-Dossman, J.P., Guio, L., García-Orjuela, D., Guzmán-Porras, J.J., Garces, K., Naranjo, A., Maradei-Anaya, S.J., & Duitama, J. (2022). Retraining and evaluation of machine learning and deep learning models for seizure classification from EEG data. *Proceedings of the IEEE ANDESCON*.

6. Goodfellow, I.J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*.

7. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *International Conference on Learning Representations (ICLR)*.

8. Blankertz, B., Müller, K.R., Krusienski, D.J., Schalk, G., Wolpaw, J.R., Schlögl, A., Pfurtscheller, G., Millán, J.R., Schröder, M., & Birbaumer, N. (2006). The BCI competition III: Validating alternative approaches to actual BCI problems. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 14(2), 153–159.

9. Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123–1134.

10. Ang, K.K., Chin, Z.Y., Zhang, H., & Guan, C. (2008). Filter bank common spatial pattern (FBCSP) in brain-computer interface. *IEEE International Joint Conference on Neural Networks (IJCNN)*, 2390–2397.
