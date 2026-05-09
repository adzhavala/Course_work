import numpy as np
import psycopg2
from collections import defaultdict
from scipy.spatial import KDTree
from config_db import DB_CONFIG
from coordinate_transforms import ra_dec_to_cartesian


class StarMatcher:
    _cached_tree = None
    _cached_triangles_data = None

    def __init__(self, db_config=None, db_password=None):
        """Initialize matcher and load triangle index."""
        if db_config:
            self.conn = psycopg2.connect(**db_config)
        elif db_password:
            self.conn = psycopg2.connect(dbname="course_work", user="postgres", password=db_password, host="localhost")
        else:
            self.conn = psycopg2.connect(**DB_CONFIG)
        self.tree = None
        self.triangles_data = []

        if StarMatcher._cached_tree is not None and StarMatcher._cached_triangles_data is not None:
            self.tree = StarMatcher._cached_tree
            self.triangles_data = StarMatcher._cached_triangles_data
        else:
            self._build_kdtree()
            if self.tree is not None and self.triangles_data:
                StarMatcher._cached_tree = self.tree
                StarMatcher._cached_triangles_data = self.triangles_data

    def _build_kdtree(self):
        """Load triangle features and build KD-Tree.
        
        KD-Tree uses only 3D ratio features [ratio1, ratio2, ratio3] to avoid
        the incompatibility between spherical and pixel orientation conventions.
        max_side_deg is stored separately for FOV filtering during query.
        """
        print("Loading triangle catalog from DB and building KD-Tree...")
        cursor = self.conn.cursor()

        try:
            # Read 3 ratios and max_side_deg; orientation is NOT read
            cursor.execute("SELECT star1_hip, star2_hip, star3_hip, ratio1, ratio2, ratio3, max_side_deg FROM star_triangles")
        except Exception:
            # Legacy: try without max_side_deg column
            cursor.execute("SELECT star1_hip, star2_hip, star3_hip, ratio1, ratio2, ratio3 FROM star_triangles")
        rows = cursor.fetchall()

        features = []
        for row in rows:
            hip1, hip2, hip3 = row[0], row[1], row[2]
            ratio1, ratio2, ratio3 = row[3], row[4], row[5]
            # max_side_deg is the 7th column (index 6) if present, else default to 180°
            max_side_deg = row[6] if len(row) > 6 and row[6] is not None else 180.0

            # 3D feature vector: only ratios, no orientation
            feature_vec = np.array([ratio1, ratio2, ratio3], dtype=np.float64)
            if not np.all(np.isfinite(feature_vec)):
                continue

            # Store HIP IDs and max_side_deg for post-search filtering
            self.triangles_data.append((hip1, hip2, hip3, max_side_deg))
            features.append(feature_vec)

        if not features:
            self.tree = None
            print("No valid triangles found for KD-Tree.")
            cursor.close()
            return

        self.tree = KDTree(features)
        print(f"KD-Tree successfully built for {len(features)} figures (3D ratio space). Ready for search!")
        cursor.close()

    def find_best_match(self, img_ratio1, img_ratio2, img_ratio3, tolerance=0.01, fov_x_deg=60.0, k=20, max_side_deg=None):
        """Find nearest matches by ratios in 3D feature space.
        
        Args:
            img_ratio1, img_ratio2, img_ratio3: Ratios from image triangle
            tolerance: Maximum tolerable error (dr1 + dr2 + dr3)
            fov_x_deg: Image field-of-view; used if max_side_deg is not provided
            k: Number of nearest neighbors to query
            max_side_deg: Optional explicit max triangle side angle (degrees)
        
        Returns:
            List of matches with structure {'star_hips': tuple, 'error': float}
        """
        if self.tree is None:
            return []

        # Query 3D KD-Tree with ratio features only
        query_pt = [img_ratio1, img_ratio2, img_ratio3]
        distances, indices = self.tree.query(query_pt, k=k)

        valid_matches = []

        for dist, idx in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if idx >= len(self.triangles_data):
                continue
            
            hip1, hip2, hip3, db_max_side_deg = self.triangles_data[idx]
            
            # FILTER: Reject DB triangles that are too large for the image FOV.
            # Use an explicit diagonal limit when provided; fall back to a wide factor.
            limit_deg = max_side_deg if max_side_deg is not None else (fov_x_deg * 1.6)
            if db_max_side_deg > limit_deg:
                continue
            
            # Compute error as sum of absolute ratio differences
            feature_vec = self.tree.data[idx]
            r1, r2, r3 = feature_vec
            error = abs(r1 - img_ratio1) + abs(r2 - img_ratio2) + abs(r3 - img_ratio3)
            
            if error <= tolerance:
                valid_matches.append({
                    'star_hips': (hip1, hip2, hip3),
                    'error': error
                })

        return valid_matches

    def find_consensus_matches(self, triangles, tolerance=0.015, per_triangle_k=20, min_support=2, fov_x_deg=60.0, max_side_deg=None):
        """Aggregate matches from multiple triangles into confidence scores."""

        if not triangles:
            return {
                'used_triangles': 0,
                'matched_triangles': 0,
                'candidates': []
            }

        aggregate = defaultdict(lambda: {'count': 0, 'error_sum': 0.0})
        matched_triangles = 0

        for t in triangles:
            # Pass fov_x_deg for max_side_deg filtering; orientation is no longer used
            matches = self.find_best_match(
                t['ratio1'],
                t['ratio2'],
                t['ratio3'],
                tolerance=tolerance,
                fov_x_deg=fov_x_deg,
                k=per_triangle_k,
                max_side_deg=max_side_deg,
            )

            if not matches:
                continue

            matched_triangles += 1
            seen_in_triangle = set()

            for m in matches:
                hips_key = tuple(sorted(m['star_hips']))
                if hips_key in seen_in_triangle:
                    continue
                seen_in_triangle.add(hips_key)

                aggregate[hips_key]['count'] += 1
                aggregate[hips_key]['error_sum'] += m['error']

        used_triangles = len(triangles)
        candidates = []

        for hips_key, stats in aggregate.items():
            count = stats['count']
            if count < min_support:
                continue
            avg_error = stats['error_sum'] / count
            support_ratio = count / used_triangles if used_triangles else 0.0
            error_quality = 1.0 / (1.0 + 100.0 * avg_error)
            confidence = 0.7 * support_ratio + 0.3 * error_quality

            candidates.append({
                'star_hips': hips_key,
                'support_count': count,
                'support_ratio': support_ratio,
                'avg_error': avg_error,
                'confidence': confidence
            })

        candidates.sort(key=lambda x: (-x['confidence'], -x['support_count'], x['avg_error']))

        return {
            'used_triangles': used_triangles,
            'matched_triangles': matched_triangles,
            'candidates': candidates
        }

    def close(self):
        self.conn.close()

    def get_star_vectors_by_hips(self, hip_ids):
        """Fetch RA/Dec from DB and convert to Cartesian unit vectors."""
        if not hip_ids:
            return {}

        cursor = self.conn.cursor()
        query = """
            SELECT hip_id, ra, dec
            FROM star_catalog
            WHERE hip_id = ANY(%s)
        """
        cursor.execute(query, (list(hip_ids),))
        rows = cursor.fetchall()
        cursor.close()

        vectors = {}
        for hip, ra_deg, dec_deg in rows:
            vectors[int(hip)] = ra_dec_to_cartesian(float(ra_deg), float(dec_deg), radius=1.0)
        return vectors

    def get_star_magnitudes_by_hips(self, hip_ids):
        """Fetch magnitudes by HIP id (lower magnitude is brighter)."""
        if not hip_ids:
            return {}

        cursor = self.conn.cursor()
        query = """
            SELECT hip_id, magnitude
            FROM star_catalog
            WHERE hip_id = ANY(%s)
        """
        cursor.execute(query, (list(hip_ids),))
        rows = cursor.fetchall()
        cursor.close()

        mags = {}
        for hip, mag in rows:
            mags[int(hip)] = float(mag)
        return mags


