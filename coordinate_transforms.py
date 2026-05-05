import math
from datetime import datetime, timezone
from typing import Iterable, Sequence, Tuple

import numpy as np


def normalize_vector(v: Sequence[float], radius: float = 1.0) -> np.ndarray:
    """Scale vector to the target sphere radius."""
    vec = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return radius * (vec / norm)


def ra_dec_to_cartesian(ra_deg: float, dec_deg: float, radius: float = 1.0) -> np.ndarray:
    """Convert celestial coordinates (RA, Dec) to Cartesian vector.

    Convention:
    x = R * cos(dec) * cos(ra)
    y = R * cos(dec) * sin(ra)
    z = R * sin(dec)
    """
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)

    x = radius * math.cos(dec_rad) * math.cos(ra_rad)
    y = radius * math.cos(dec_rad) * math.sin(ra_rad)
    z = radius * math.sin(dec_rad)
    return np.array([x, y, z], dtype=np.float64)


def cartesian_to_ra_dec(v: Sequence[float]) -> Tuple[float, float]:
    """Convert Cartesian vector to (RA, Dec) in degrees."""
    x, y, z = normalize_vector(v, radius=1.0)
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(z))
    return ra, dec


def angular_separation_deg(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Angular distance between two vectors in degrees."""
    a = normalize_vector(v1, radius=1.0)
    b = normalize_vector(v2, radius=1.0)
    c = float(np.dot(a, b))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def solve_vector_from_angular_constraints(
    star_vectors: Iterable[Sequence[float]],
    alphas_deg: Sequence[float],
    radius: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve pointing vector from equations (r, n_i) = R^2 * cos(alpha_i).

    This directly matches the board derivation:
    A * r = b, where each row of A is star vector n_i, and
    b_i = R^2 * cos(alpha_i).

    Works with 3 or more equations via least squares.
    Returns:
    - r_on_sphere: normalized solution on radius R
    - residuals: A*r_on_sphere - b
    """
    A = np.asarray(list(star_vectors), dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("star_vectors must be an iterable of 3D vectors")

    alpha = np.asarray(alphas_deg, dtype=np.float64)
    if len(alpha) != A.shape[0]:
        raise ValueError("alphas_deg length must match number of star vectors")
    if A.shape[0] < 3:
        raise ValueError("At least 3 stars are required")

    b = (radius ** 2) * np.cos(np.radians(alpha))

    # Least squares keeps this robust when equations are noisy.
    r_raw, *_ = np.linalg.lstsq(A, b, rcond=None)

    r_on_sphere = normalize_vector(r_raw, radius=radius)
    residuals = A @ r_on_sphere - b
    return r_on_sphere, residuals


def solve_vector_from_ra_dec(
    stars_ra_dec_deg: Iterable[Tuple[float, float]],
    alphas_deg: Sequence[float],
    radius: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: build star vectors from RA/Dec and solve."""
    vectors = [ra_dec_to_cartesian(ra, dec, radius=radius) for ra, dec in stars_ra_dec_deg]
    return solve_vector_from_angular_constraints(vectors, alphas_deg, radius=radius)


def julian_date(dt_utc: datetime) -> float:
    """Convert UTC datetime to Julian Date."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt_utc.astimezone(timezone.utc)

    year = dt_utc.year
    month = dt_utc.month
    day = dt_utc.day

    frac_day = (
        dt_utc.hour
        + dt_utc.minute / 60.0
        + dt_utc.second / 3600.0
        + dt_utc.microsecond / 3_600_000_000.0
    ) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4
    jd = (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
        + frac_day
    )
    return jd


def greenwich_sidereal_time_deg(dt_utc: datetime) -> float:
    """Compute Greenwich mean sidereal time in degrees."""
    jd = julian_date(dt_utc)
    t = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - (t * t * t) / 38710000.0
    return gmst % 360.0


def normalize_angle_deg_signed(angle_deg: float) -> float:
    """Normalize angle to [-180, 180) degrees."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def calculate_geographic_coordinates(
    rotation_matrix: Sequence[Sequence[float]],
    utc_time: datetime,
    zenith_vector: Sequence[float],
) -> Tuple[float, float, float, float, float]:
    """Compute geocentric latitude/longitude from camera-frame zenith.

    rotation_matrix must map camera-frame vectors into the inertial frame.
    Returns: (lat_deg, lon_deg, ra_z_deg, dec_z_deg, gmst_deg)
    """
    rot = np.asarray(rotation_matrix, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError("rotation_matrix must be 3x3")

    z_cam = normalize_vector(zenith_vector, radius=1.0)
    z_eci = normalize_vector(rot @ z_cam, radius=1.0)

    ra_deg, dec_deg = cartesian_to_ra_dec(z_eci)
    gmst_deg = greenwich_sidereal_time_deg(utc_time)

    # Geocentric latitude equals zenith declination in the inertial frame.
    lat_deg = dec_deg
    # Longitude: lambda = alpha - GMST, normalized to [-180, 180).
    lon_deg = normalize_angle_deg_signed(ra_deg - gmst_deg)

    return lat_deg, lon_deg, ra_deg, dec_deg, gmst_deg


def eci_to_ecef(v_eci: Sequence[float], dt_utc: datetime, rotation_sign: float = 1.0) -> np.ndarray:
    """Rotate vector from Earth-centered inertial frame to Earth-fixed frame."""
    theta = rotation_sign * math.radians(greenwich_sidereal_time_deg(dt_utc))
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rot @ np.asarray(v_eci, dtype=np.float64)


def ecef_to_lat_lon_deg(v_ecef: Sequence[float]) -> Tuple[float, float]:
    """Convert ECEF vector to geocentric latitude/longitude in degrees."""
    x, y, z = normalize_vector(v_ecef, radius=1.0)
    lat = math.degrees(math.asin(z))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def pixel_to_camera_ray(center_xy: Sequence[float], image_shape: Sequence[int], fov_x_deg: float) -> np.ndarray:
    """Convert pixel location to a unit ray in camera frame (pinhole model)."""
    height, width = image_shape[:2]
    cx = width / 2.0
    cy = height / 2.0

    f = (width / 2.0) / math.tan(math.radians(fov_x_deg / 2.0))
    x = (center_xy[0] - cx) / f
    y = (center_xy[1] - cy) / f

    # In image coordinates +y is down, so invert y for right-handed camera frame.
    vec = np.array([x, -y, 1.0], dtype=np.float64)
    return normalize_vector(vec, radius=1.0)


def estimate_rotation_kabsch(reference_vectors: Sequence[Sequence[float]], observed_vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Estimate rotation matrix R such that observed ~= R * reference."""
    ref = np.asarray(reference_vectors, dtype=np.float64)
    obs = np.asarray(observed_vectors, dtype=np.float64)

    if ref.shape != obs.shape or ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError("reference_vectors and observed_vectors must be Nx3 with same shape")
    if ref.shape[0] < 2:
        raise ValueError("At least 2 vector correspondences are required")

    h = ref.T @ obs
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Enforce proper rotation (determinant +1), avoid reflections.
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    return r


def angular_residuals_deg(reference_vectors: Sequence[Sequence[float]], observed_vectors: Sequence[Sequence[float]], rotation: np.ndarray) -> np.ndarray:
    """Per-pair angular residuals in degrees for observed vs R*reference."""
    ref = np.asarray(reference_vectors, dtype=np.float64)
    obs = np.asarray(observed_vectors, dtype=np.float64)

    modeled = (rotation @ ref.T).T
    out = []
    for m, o in zip(modeled, obs):
        c = float(np.dot(normalize_vector(m), normalize_vector(o)))
        c = max(-1.0, min(1.0, c))
        out.append(math.degrees(math.acos(c)))
    return np.asarray(out, dtype=np.float64)
