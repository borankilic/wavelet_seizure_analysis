import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt


# ============================================================================
# WAVELET BASIS GENERATION
# ============================================================================

def get_haar_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Haar wavelet scaling and wavelet coefficients.
    
    Returns:
        h_phi: Scaling function coefficients (lowpass)
        h_psi: Wavelet function coefficients (highpass)
    """
    h_phi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Lowpass filter
    h_psi = np.array([1/np.sqrt(2), -1/np.sqrt(2)])  # Highpass filter
    return h_phi, h_psi


def get_db2_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Daubechies-2 (db2) wavelet coefficients.
    Also known as D4 - has 4 coefficients.
    
    Returns:
        h_phi: Scaling function coefficients (lowpass)
        h_psi: Wavelet function coefficients (highpass)
    """
    # Daubechies D4 scaling coefficients
    h_phi = np.array([
        (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
        (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
        (1 - np.sqrt(3)) / (4 * np.sqrt(2))
    ])
    
    # Wavelet coefficients using QMF relationship
    h_psi = np.array([h_phi[3], -h_phi[2], h_phi[1], -h_phi[0]])
    
    return h_phi, h_psi


def get_sym4_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Symlet-4 wavelet coefficients.
    Symlets are nearly symmetric variants of Daubechies wavelets.
    
    Returns:
        h_phi: Scaling function coefficients (lowpass)
        h_psi: Wavelet function coefficients (highpass)
    """
    # Symlet-4 has 8 coefficients
    h_phi = np.array([
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.0322231006040427
    ])
    
    # Wavelet coefficients using QMF relationship
    N = len(h_phi)
    h_psi = np.array([(-1)**n * h_phi[N-1-n] for n in range(N)])
    
    return h_phi, h_psi


def get_wavelet_filters(wavelet_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get filter coefficients for specified wavelet family.
    
    Args:
        wavelet_name: Name of wavelet ('haar', 'db2', 'sym4')
    
    Returns:
        h_phi: Scaling function coefficients (lowpass)
        h_psi: Wavelet function coefficients (highpass)
    """
    wavelet_dict = {
        'haar': get_haar_coefficients,
        'db2': get_db2_coefficients,
        'sym4': get_sym4_coefficients
    }
    
    if wavelet_name.lower() not in wavelet_dict:
        raise ValueError(f"Unknown wavelet: {wavelet_name}. Choose from {list(wavelet_dict.keys())}")
    
    return wavelet_dict[wavelet_name.lower()]()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def pad_to_power_of_2(signal: np.ndarray) -> np.ndarray:
    """
    Pad signal with zeros to make its length a power of 2.
    
    Args:
        signal: Input signal
    
    Returns:
        Padded signal with length 2^N
    """
    N = len(signal)
    J = int(np.ceil(np.log2(N)))
    target_length = 2 ** J
    
    if N < target_length:
        padded = np.zeros(target_length)
        padded[:N] = signal
        return padded
    
    return signal


def generate_basis_functions(h_phi: np.ndarray, h_psi: np.ndarray, 
                            j: int, k: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate scaling and wavelet basis functions at scale j and translation k.
    
    Args:
        h_phi: Scaling function coefficients
        h_psi: Wavelet function coefficients
        j: Scale parameter
        k: Translation parameter
        N: Length of signal
    
    Returns:
        phi: Scaling function samples
        psi: Wavelet function samples
    """
    scale_factor = 2 ** (j / 2)
    x = np.arange(N) / (2 ** j) - k
    
    # Simple approximation: use coefficients as basis
    phi = np.zeros(N)
    psi = np.zeros(N)
    
    for i, coef in enumerate(h_phi):
        idx = int(k * 2**j + i)
        if 0 <= idx < N:
            phi[idx] = coef * scale_factor
    
    for i, coef in enumerate(h_psi):
        idx = int(k * 2**j + i)
        if 0 <= idx < N:
            psi[idx] = coef * scale_factor
    
    return phi, psi


# ============================================================================
# NAIVE DWT IMPLEMENTATION
# ============================================================================

def naive_dwt(signal: np.ndarray, wavelet_name: str = 'haar', 
              j0: int = 0, max_level: int = None) -> Dict:
    """
    Compute Discrete Wavelet Transform using direct inner product calculation.
    
    This implements Equations (7-137) and (7-138) from the document:
    - T_phi(0,k) = (1/√N) * Σ f(x) * φ(x)
    - T_psi(j,k) = (1/√N) * Σ f(x) * ψ_j,k(x)
    
    Args:
        signal: Input signal
        wavelet_name: Name of wavelet basis ('haar', 'db2', 'sym4')
        j0: Starting scale (default 0)
        max_level: Maximum decomposition level
    
    Returns:
        Dictionary containing approximation and detail coefficients
    """
    # Pad signal to power of 2
    signal = pad_to_power_of_2(signal)
    N = len(signal)
    J = int(np.log2(N))
    
    if max_level is None:
        max_level = J - 1
    
    # Get wavelet filters
    h_phi, h_psi = get_wavelet_filters(wavelet_name)
    
    # Normalization factor
    norm = 1 / np.sqrt(N)
    
    # Initialize result dictionary
    result = {
        'approximation': {},
        'details': {},
        'j0': j0,
        'J': J,
        'N': N
    }
    
    # Compute approximation coefficients at scale j0
    # T_phi(j0, k) for k = 0, 1, ..., 2^j0 - 1
    num_coeffs_j0 = 2 ** j0 if j0 > 0 else 1
    approx_coeffs = np.zeros(num_coeffs_j0)
    
    for k in range(num_coeffs_j0):
        phi, _ = generate_basis_functions(h_phi, h_psi, j0, k, N)
        approx_coeffs[k] = norm * np.sum(signal * phi)
    
    result['approximation'][j0] = approx_coeffs
    
    # Compute detail coefficients at scales j0, j0+1, ..., max_level
    for j in range(j0, min(max_level + 1, J)):
        num_coeffs = 2 ** j
        detail_coeffs = np.zeros(num_coeffs)
        
        for k in range(num_coeffs):
            _, psi = generate_basis_functions(h_phi, h_psi, j, k, N)
            detail_coeffs[k] = norm * np.sum(signal * psi)
        
        result['details'][j] = detail_coeffs
    
    return result


def naive_idwt(coefficients: Dict, wavelet_name: str = 'haar') -> np.ndarray:
    """
    Compute Inverse DWT from wavelet coefficients.
    
    Implements Equation (7-136):
    f(x) = (1/√N) * [Σ T_phi * φ + ΣΣ T_psi * ψ]
    
    Args:
        coefficients: Dictionary from naive_dwt
        wavelet_name: Name of wavelet basis
    
    Returns:
        Reconstructed signal
    """
    N = coefficients['N']
    j0 = coefficients['j0']
    norm = 1 / np.sqrt(N)
    
    h_phi, h_psi = get_wavelet_filters(wavelet_name)
    
    # Initialize reconstructed signal
    reconstructed = np.zeros(N)
    
    # Add approximation component
    approx_coeffs = coefficients['approximation'][j0]
    for k, coeff in enumerate(approx_coeffs):
        phi, _ = generate_basis_functions(h_phi, h_psi, j0, k, N)
        reconstructed += norm * coeff * phi
    
    # Add detail components
    for j in sorted(coefficients['details'].keys()):
        detail_coeffs = coefficients['details'][j]
        for k, coeff in enumerate(detail_coeffs):
            _, psi = generate_basis_functions(h_phi, h_psi, j, k, N)
            reconstructed += norm * coeff * psi
    
    return reconstructed


# ============================================================================
# FAST WAVELET TRANSFORM (FWT) IMPLEMENTATION
# ============================================================================

def convolve_and_downsample(signal: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
    """
    Convolve signal with filter and downsample by 2.
    
    This implements the core operation in Equations (7-143) and (7-144):
    - Convolve with time-reversed filter h(-n)
    - Keep only even-indexed samples (downsampling by 2)
    
    Args:
        signal: Input signal
        filter_coeffs: Filter coefficients (will be time-reversed)
    
    Returns:
        Downsampled convolution result
    """
    # CONVOLUTION: Flip filter coefficients for convolution (implements h(-n))
    flipped_filter = filter_coeffs[::-1]
    
    # Perform convolution with 'same' mode to maintain alignment
    convolved = np.convolve(signal, flipped_filter, mode='same')
    
    # DOWNSAMPLING: Keep only even-indexed samples (downsampling by 2)
    downsampled = convolved[::2]
    
    return downsampled


def upsample_and_convolve(signal: np.ndarray, filter_coeffs: np.ndarray, 
                          target_length: int) -> np.ndarray:
    """
    Upsample signal by 2 and convolve with filter.
    
    This implements the synthesis/reconstruction operation:
    - Insert zeros between samples (upsampling by 2)
    - Convolve with synthesis filter
    
    Args:
        signal: Input signal
        filter_coeffs: Synthesis filter coefficients
        target_length: Desired output length
    
    Returns:
        Upsampled and convolved result
    """
    # UPSAMPLING: Insert zeros between samples (implements Equation 7-148)
    upsampled = np.zeros(len(signal) * 2)
    upsampled[::2] = signal
    
    # CONVOLUTION: Convolve with synthesis filter
    convolved = np.convolve(upsampled, filter_coeffs, mode='same')
    
    # Trim or pad to target length
    if len(convolved) > target_length:
        convolved = convolved[:target_length]
    elif len(convolved) < target_length:
        padded = np.zeros(target_length)
        padded[:len(convolved)] = convolved
        convolved = padded
    
    return convolved


def fwt(signal: np.ndarray, wavelet_name: str = 'haar', 
        level: int = None) -> Dict:
    """
    Fast Wavelet Transform using filter banks.
    
    Implements the iterative filter bank structure from Figure 7.24:
    - Each stage splits the approximation into new approximation and detail
    - Uses convolution with time-reversed filters followed by downsampling
    
    Args:
        signal: Input signal
        wavelet_name: Name of wavelet basis
        level: Number of decomposition levels (default: maximum possible)
    
    Returns:
        Dictionary with approximation and detail coefficients at each level
    """
    # Pad to power of 2
    signal = pad_to_power_of_2(signal)
    N = len(signal)
    J = int(np.log2(N))
    
    if level is None:
        level = J
    level = min(level, J)
    
    # Get filter coefficients (h_phi = lowpass, h_psi = highpass)
    h_phi, h_psi = get_wavelet_filters(wavelet_name)
    
    # Initialize result
    result = {
        'approximation': {},
        'details': {},
        'level': level,
        'N': N,
        'wavelet': wavelet_name
    }
    
    # Start with signal as highest resolution approximation
    current_approx = signal.copy()
    
    # Iterate through decomposition levels (implements Figure 7.24a)
    for j in range(level):
        # ANALYSIS FILTER BANK:
        
        # Lowpass branch: convolve with h_phi(-n) and downsample
        # This produces approximation coefficients at the next coarser scale
        new_approx = convolve_and_downsample(current_approx, h_phi)
        
        # Highpass branch: convolve with h_psi(-n) and downsample
        # This produces detail coefficients at the current scale
        detail = convolve_and_downsample(current_approx, h_psi)
        
        # Store coefficients
        scale = J - j - 1  # Scale in terms of J-j notation
        result['details'][scale] = detail
        
        # Update current approximation for next iteration
        current_approx = new_approx
    
    # Store final approximation coefficients
    result['approximation'][J - level] = current_approx
    
    return result


def ifwt(coefficients: Dict) -> np.ndarray:
    """
    Inverse Fast Wavelet Transform.
    
    Reconstructs signal from wavelet coefficients using synthesis filter bank.
    
    Args:
        coefficients: Dictionary from fwt
    
    Returns:
        Reconstructed signal
    """
    N = coefficients['N']
    level = coefficients['level']
    J = int(np.log2(N))
    wavelet_name = coefficients['wavelet']
    
    # Get synthesis filters (time-reversed analysis filters for orthonormal wavelets)
    h_phi, h_psi = get_wavelet_filters(wavelet_name)
    
    # For orthonormal wavelets, synthesis filters are time-reversed analysis filters
    g_phi = h_phi[::-1]  # Lowpass synthesis
    g_psi = h_psi[::-1]  # Highpass synthesis
    
    # Start with coarsest approximation
    start_scale = J - level
    reconstructed = coefficients['approximation'][start_scale]
    
    # Iteratively reconstruct through each level
    for j in range(level):
        scale = start_scale + j
        target_length = len(reconstructed) * 2
        
        # SYNTHESIS FILTER BANK:
        
        # Upsample and convolve approximation with lowpass synthesis filter
        approx_contribution = upsample_and_convolve(reconstructed, g_phi, target_length)
        
        # Upsample and convolve detail with highpass synthesis filter
        detail = coefficients['details'][scale]
        detail_contribution = upsample_and_convolve(detail, g_psi, target_length)
        
        # Combine both contributions
        reconstructed = approx_contribution + detail_contribution
    
    return reconstructed[:N]


# ============================================================================
# VISUALIZATION AND TESTING
# ============================================================================

def print_dwt_results(coeffs: Dict, method: str = "Naive DWT"):
    """Print DWT coefficients in a readable format."""
    print(f"\n{method} Results:")
    print(f"Signal length: {coeffs['N']}")
    print(f"\nApproximation coefficients:")
    for scale, values in sorted(coeffs['approximation'].items()):
        print(f"  Scale {scale}: {values}")
    print(f"\nDetail coefficients:")
    for scale, values in sorted(coeffs['details'].items()):
        print(f"  Scale {scale}: {values}")


def demo_dwt():
    """Demonstrate DWT implementations with example from the document."""
    print("=" * 70)
    print("DISCRETE WAVELET TRANSFORM DEMONSTRATION")
    print("=" * 70)
    
    # Example from the document: [1, 4, 3, 0]
    signal = np.array([1, 4, 3, 0])
    print(f"\nOriginal signal: {signal}")
    
    # Test Naive DWT
    print("\n" + "-" * 70)
    print("NAIVE DWT (Direct inner product calculation)")
    print("-" * 70)
    naive_coeffs = naive_dwt(signal, 'haar', j0=0, max_level=1)
    print_dwt_results(naive_coeffs, "Naive DWT")
    
    # Test reconstruction
    reconstructed_naive = naive_idwt(naive_coeffs, 'haar')
    print(f"\nReconstructed signal: {reconstructed_naive}")
    print(f"Reconstruction error: {np.max(np.abs(signal - reconstructed_naive)):.10f}")
    
    # Test Fast Wavelet Transform
    print("\n" + "-" * 70)
    print("FAST WAVELET TRANSFORM (Filter bank implementation)")
    print("-" * 70)
    fwt_coeffs = fwt(signal, 'haar', level=2)
    print_dwt_results(fwt_coeffs, "FWT")
    
    # Test FWT reconstruction
    reconstructed_fwt = ifwt(fwt_coeffs)
    print(f"\nReconstructed signal: {reconstructed_fwt}")
    print(f"Reconstruction error: {np.max(np.abs(signal - reconstructed_fwt)):.10f}")
    
    # Test with other wavelets
    print("\n" + "=" * 70)
    print("TESTING OTHER WAVELET FAMILIES")
    print("=" * 70)
    
    test_signal = np.array([1, 2, 4, 8, 5, 3, 2, 1])
    
    for wavelet in ['haar', 'db2', 'sym4']:
        print(f"\n{wavelet.upper()} Wavelet:")
        print("-" * 40)
        coeffs = fwt(test_signal, wavelet, level=2)
        reconstructed = ifwt(coeffs)
        error = np.max(np.abs(test_signal - reconstructed))
        print(f"Approximation: {coeffs['approximation'][1]}")
        print(f"Details scale 1: {coeffs['details'][1]}")
        print(f"Details scale 2: {coeffs['details'][2]}")
        print(f"Reconstruction error: {error:.10f}")


if __name__ == "__main__":
    demo_dwt()