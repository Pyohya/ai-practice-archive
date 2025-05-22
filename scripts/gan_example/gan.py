import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
NOISE_DIM = 10       # Dimension of the noise vector for the generator
DATA_DIM = 1         # Dimension of the data to generate (1D data)
REAL_DATA_MEAN = 5.0 # Mean of the real data distribution
REAL_DATA_STDDEV = 0.5 # Std dev of the real data distribution

# --- 1. Generator Model ---
def build_generator(noise_dim=NOISE_DIM, output_dim=DATA_DIM):
    model = Sequential(name="Generator")
    model.add(Dense(32, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8)) # Helps stabilize training
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='linear')) # Output layer for 1D data
    
    #print("--- Generator Summary ---")
    #model.summary()
    return model

# --- 2. Discriminator Model ---
def build_discriminator(data_dim=DATA_DIM):
    model = Sequential(name="Discriminator")
    model.add(Dense(64, input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid')) # Output layer: probability (real or fake)
    
    #print("--- Discriminator Summary ---")
    #model.summary()
    return model

# --- 3. Combined GAN Model ---
def build_gan(generator, discriminator):
    # Freeze discriminator weights during GAN training
    discriminator.trainable = False
    
    model = Sequential(name="GAN")
    model.add(generator)
    model.add(discriminator)
    
    #print("--- GAN Summary ---")
    #model.summary()
    return model

# --- 4. Prepare Real Data ---
def get_real_samples(batch_size, mean=REAL_DATA_MEAN, stddev=REAL_DATA_STDDEV):
    # Generate real samples from a normal distribution
    real_data = np.random.normal(loc=mean, scale=stddev, size=(batch_size, DATA_DIM))
    return real_data

# --- 5. Training Loop ---
def train_gan(generator, discriminator, gan_model, epochs=10000, batch_size=64, sample_interval=1000):
    # Optimizers
    # Using tf.keras.optimizers.Adam directly as Keras 3 default
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    # Compile discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    
    # Compile GAN (with frozen discriminator, so only generator is trained)
    gan_model.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    # Adversarial ground truths
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    print(f"Starting GAN training for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Train Discriminator ---
        # Select a random batch of real samples
        real_samples = get_real_samples(batch_size)
        
        # Generate a batch of fake samples
        noise_d = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        fake_samples = generator.predict(noise_d, verbose=0)
        
        # Train the discriminator (real classified as 1 and fake as 0)
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- Train Generator ---
        # Generate noise as input for the generator
        noise_g = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        
        # Train the generator (via the GAN model, try to make discriminator classify fakes as real)
        g_loss = gan_model.train_on_batch(noise_g, real_labels)

        # --- Logging ---
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}/{epochs} | D loss: {d_loss[0]:.4f} (Acc: {100*d_loss[1]:.2f}%) | G loss: {g_loss:.4f}")
            # Generate a few samples to see progress
            generate_and_show_samples(generator, epoch, mean_real=REAL_DATA_MEAN)

    print("GAN Training Finished.")

# --- Helper function to generate and show samples ---
def generate_and_show_samples(generator, epoch, n_samples=5, mean_real=REAL_DATA_MEAN):
    noise = np.random.normal(0, 1, (n_samples, NOISE_DIM))
    generated_samples = generator.predict(noise, verbose=0)
    print(f"--- Samples at epoch {epoch} (Real mean ~{mean_real:.2f}) ---")
    for i in range(n_samples):
        print(f"Sample {i+1}: {generated_samples[i][0]:.4f}")
    print("------------------------------------")


# --- 6. Main Execution Block ---
if __name__ == '__main__':
    # Build and compile the discriminator
    discriminator = build_discriminator()
    # No compilation here, will be compiled in train_gan or when used standalone

    # Build the generator
    generator = build_generator()
    # No compilation here, will be compiled as part of GAN or when used standalone

    # Build and compile the GAN
    gan_model = build_gan(generator, discriminator)
    # No compilation here, will be compiled in train_gan

    print("--- Initializing GAN ---")
    print(f"Noise dimension: {NOISE_DIM}")
    print(f"Data dimension: {DATA_DIM}")
    print(f"Target real data mean: {REAL_DATA_MEAN}, stddev: {REAL_DATA_STDDEV}")
    print("--- Generator Model ---")
    generator.summary()
    print("\n--- Discriminator Model ---")
    discriminator.summary()
    print("\n--- Combined GAN Model ---")
    gan_model.summary()
    
    # Train the GAN
    # Adjusted epochs for a reasonable demonstration run
    train_gan(generator, discriminator, gan_model, epochs=1000, batch_size=64, sample_interval=200)

    # Generate some final samples
    print("\n--- Generating final samples after training ---")
    noise = np.random.normal(0, 1, (10, NOISE_DIM))
    final_samples = generator.predict(noise, verbose=0)
    for i, sample in enumerate(final_samples):
        print(f"Final Sample {i+1}: {sample[0]:.4f}")

    # Optional: Plot a histogram of generated samples vs real samples
    # Needs matplotlib: pip install matplotlib
    num_plot_samples = 1000
    real_plot_samples = get_real_samples(num_plot_samples)
    noise_plot = np.random.normal(0, 1, (num_plot_samples, NOISE_DIM))
    generated_plot_samples = generator.predict(noise_plot, verbose=0)

    plt.figure(figsize=(10, 6))
    plt.hist(real_plot_samples.flatten(), bins=50, alpha=0.6, label=f'Real Data (Mean: {REAL_DATA_MEAN})', color='blue', density=True)
    plt.hist(generated_plot_samples.flatten(), bins=50, alpha=0.6, label='Generated Data', color='orange', density=True)
    plt.title('Distribution of Real vs. Generated Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('scripts/gan_example/gan_generated_data_distribution.png') # Save plot in the example directory
    print(f"\nSaved plot of data distributions to scripts/gan_example/gan_generated_data_distribution.png")
    # plt.show() # Uncomment if running in an environment that supports GUI pop-ups
