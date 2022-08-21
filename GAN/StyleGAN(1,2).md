# StyleGAN

## StyleGAN1: A Style-Based Generator Architecture for Generative Adversarial Networks

### 01. Style-based generator

1) Overview

- StyleGAN1 maps $Z$ latent space to $W$.
- Not explicitly employing the latent code z, mapping network $f$ is introduced, thus 512-dimension mapped vector $w$ is used for style transfer
- AdaIN layer for progressive style transfer
- Styles are overwritten at every normalization layer
- Add finer details to coarse image
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805980-50e5d1d7-db2e-4632-b968-5bbe6398bda0.png)
    

2) Latent vector mapping for disentanglement

- Latent vector z directly sampled from $Z$ follows the train data distribution, thus entangled.
- There are some forbidden combinations that does not exist in training dataset → Mapping from Z to features will cause distortion of the feature space.
- Mapping from W to features can resolve this problem.

3) AdaIN for style transfer

- Instance normalization
    - $CIN(x;s) = \gamma^s{x-\mu(x)\over \sigma(x)} + \beta^s$
        - for every style s
- Adaptively calculate affine parameter based on feature statics: AdaIN
    - $AdaIN(x;y) = \sigma(y) {x-\mu(x) \over \sigma(x)} + \mu(y)$
    - normalize by feature statistics, thus AdaIN overwrites the previous style

4) Explicit noise inputs for stochastic variation

- Generator network uses **learned constant input for generator, instead of latent vector**
- Noise input are added for stochastic details
- Added after convolution

### 02. PGGAN → StyleGAN

**Details of architecture & experiment**

![Untitled](https://user-images.githubusercontent.com/75057952/185805960-0490f762-4e8f-4741-aa98-486487750413.png)

- Baseline: PGGAN [A]
- Upsampling & Downsampling method modification & longer training → tuning for better performance [B]
- Mapping networks & AdaIN operations(Novelty) [C]
    - Mapping networks: Z → W
    - AdaIN
- Traditional input removal [D]
    - astonishingly improves the performance of generator
- Noise inputs are added explicitly [E]
- Mixing regularization [F]

### 03. Properties of style-based generator

1) Style mixing

- Mixing regularization decorrelates neighboring styles
- Run to latent codes z1, z2 and map to w1, w2
- Mixing regularization prevents adjacent layer being highly correlated
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805961-99342458-c033-4804-932f-56e9e1aa7193.png)
    

2) Stochastic variation

- Per-pixel noise addition leads to stochastic variation
- Noise inputs only affect stochastic aspects, and overall composition are identity intact
- Inserting noise in to images → check for standard deviation per pixel
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805964-e277b75d-9ab9-4e64-a170-94b5823b7549.png)
    
- Adding noise to all layers(a), no noise(b)
- Noise in fine layers(c) → finer curls & background
- Noise in coarse layers(d) → large scale curls & background
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805965-21d20b17-b530-45c8-beed-f0b490fe3a1d.png)
    

3) Separation of global effects from stochasticity

- Style vector(usually identical) → applied globally
- Global effects(pose, bg style, lighting) can be controlled coherently, while stochastic variations by noise inputs are independent
- **Spatial-invariant statistics such as gram-matrix & channel-wise mean, variance → encoding the ‘style’**
- Spatially varying features → encoding a specific instance

### 4. Disentanglement studies

1) Perceptual path length(PPL)

- Latent space is ‘disentangled’ when it consists of linear subspaces
- Each linear subspaces control one factor of variation
- Sampling probability of each combinations in $Z$ follows training data distribution → intermediate latent space $W$ and mapping function $f(z)$ forces each variation factors to be **unwarped**
- **Metric for disentanglement quantification are required.**
    - PPL: How network output $\phi(x)$ is perturbed while interpolating $x$ between to latent vectors $w_1$ and $w_2$
        
        ![Untitled](https://user-images.githubusercontent.com/75057952/185805966-73dfa91b-a497-4a35-b463-b3af03d23814.png)
        
        - PPL is space $Z, W$ $\uarr$
    - **Small PPL : Generator network is robust to perturbation while interpolating two latent vectors**
        - i.e. each features are less correlated & thus disentangled.

2) Linear separability

- Train auxiliary classifiers(e.g. classify whether male or female)
- Fit linear SVM to predict the label based on **latent code**
- Separability score = $\exp(\sum_i{H(Y_i|X_i)})$ for various features i(i are attributes, including hair color and etc.)
- Conditional entropy minimizes if latent space is disentangled into linear subspaces, due to high SVM classifier performance
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805967-74b474fa-d836-4f58-81c6-e7a2b47eca37.png)
    

## StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN

### 1. StyleGAN2 overview

- Removing Artifacts
    - Droplet-like artifacts, related to instance normalization
        
        ![Untitled](https://user-images.githubusercontent.com/75057952/185805969-8d88c112-ab13-496d-9718-1f902d6775ee.png)
        
        - Redesign normalization network
    - Phase artifact, related to progressive growing
        
        ![Untitled](https://user-images.githubusercontent.com/75057952/185805971-ffcf4e82-545b-4908-a323-a42e9933653e.png)
        
        - Alternative network architecture without **topology change at training time**
- Better performance
    - Smoothness of generator& image quality
        - Path length regularization → Interestingly benefits inverse problem

### 2. Removing normalization artifacts

1) Normalization artifacts

- Blob-like artifacts that resemble droplet are observed
- Usually starts to occur at 64x64 resolution → progressively stronger at larger resolutions
- Cause: **AdaIN** operation that normalized mean & variance for every instance
    - AdaIN destroys inter-instance features relative to each other
    - Droplets might be caused by dominating statistics(especially mean), causing spike signals

2) Instance normalization revisited: Weight demodulation

- AdaIN = normalization followed by modulation
- AdaIN$(x;y)$ = $\sigma(y) {x-\mu(x) \over \sigma(x)} + \mu(y)$
- **Inter-channel scale relationship might collapse by mean normalization!**
- Instead of normalizing feature maps, demodulate the **weight of conv layer** → normalized convolution output explicitly
- Modulation step: scale weight by $s_i: w_{ijk}’ = s_i \cdot w_{ijk}$
- Demodulation step: normalize standard deviation, not mean: $w_{ijk}’’ ={ {w'}_{ijk} \over {\sqrt{\sum_{i,k} {w'}_{ijk}^2 + \epsilon}}}$

### 3. Image quality and generator smoothness

1) PPL score & semantic consistency

- PPL score was first introduced for smoothness of mapping metric
- Interestingly, low per-image PPL score has high correlation with image quality… WHY?
    - Researchers hypothesizes that **broken image, which is easily distinguishable by discriminator are not favored**
    - Thus generator finds a way to effectively stretch latent vector space that yields good images.
    - Corrupted image-generating latent space are ‘squeezed’ → high PPL due to rapid generated image change near that space
    - For long-term impact, accumulated distortion of topology might cause impaired training dynamics

2) Lazy regularization

- K iterations with gradient of **main loss →** 1 iteration with gradient of regularization loss
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805973-00a86512-5480-4c7e-9795-e3125b99b766.png)
    
- Computationally efficient

3) Path length regularization

- vector $a$ indicates **global average value of mapped vector change**
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805974-9c596fe8-6ecd-443d-9cee-940832d5aa47.png)
    
- Norm of $J_w^Ty$ = size of transformed vector of y
- We expect that the change of mapped vector(kind of path length) are regularized under $a$.
- Loss is minimized when $J_W^T = U\tilde{\Sigma}V^T$, where $\Sigma = {a \over \sqrt{L}}I$

### 4. Progressive growing revisited

Progressive growing leads to phase artifacts

- some features are fixed at high frequency state at the early stage of progressive training
- high frequency = high in contrast with adjacent pixels
- high-frequency fixed features at low resolution early stages are barely modified during high resolution stage
- Alternative network architectures for progressive growing
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805975-a77c3d19-0317-49ac-b194-5c765c20bbb1.png)
    
    - MSG-GAN
    - Input/output skips
    - Residual Nets
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/185805976-54b5fb46-a521-4e41-87c8-711b01377f47.png)
    
    - Residual discriminator works well, while residual generator works poorly

### 5. Projection of images to latent space

Inverse problem intuition

- Inverse problem(e.g. source localization) is generative problem
- To find latent vector $w$ that generates target image $x$, we utilized trained generator networks(StyleGAN, StyleGAN2)

![Untitled](https://user-images.githubusercontent.com/75057952/185805977-6c52040c-783c-4c98-8d5b-363953a6a3f6.png)

![Untitled](https://user-images.githubusercontent.com/75057952/185805979-f5570884-3c4c-447c-bb32-c312b2632783.png)