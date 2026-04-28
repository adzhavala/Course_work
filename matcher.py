import numpy as np
import psycopg2
from collections import defaultdict
from scipy.spatial import KDTree
from config_db import DB_CONFIG
from coordinate_transforms import ra_dec_to_cartesian


class StarMatcher:
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

        self._build_kdtree()

    def _build_kdtree(self):
        """Load triangle features and build KD-Tree."""
        print("Loading triangle catalog from DB and building KD-Tree...")
        cursor = self.conn.cursor()

        try:
            cursor.execute("SELECT star1_hip, star2_hip, star3_hip, ratio1, ratio2, orientation, ratio3 FROM star_triangles")
        except Exception:
            cursor.execute("SELECT star1_hip, star2_hip, star3_hip, ratio1, ratio2 FROM star_triangles")
        rows = cursor.fetchall()

        features = []
        for row in rows:
            hip1, hip2, hip3 = row[0], row[1], row[2]
            ratio1, ratio2 = row[3], row[4]
            ratio3 = row[6] if len(row) > 6 and row[6] is not None else (ratio1 / ratio2 if ratio2 else 0.0)
            orientation = row[5] if len(row) > 5 and row[5] is not None else 0.0

            feature_vec = np.array([ratio1, ratio2, ratio3, orientation], dtype=np.float64)
            if not np.all(np.isfinite(feature_vec)):
                continue

            self.triangles_data.append((hip1, hip2, hip3, orientation))
            features.append(feature_vec)

        if not features:
            self.tree = None
            print("No valid triangles found for KD-Tree.")
            cursor.close()
            return

        self.tree = KDTree(features)
        print(f"KD-Tree successfully built for {len(features)} figures. Ready for search!")
        cursor.close()

    def find_best_match(self, img_ratio1, img_ratio2, img_ratio3, img_orientation, tolerance=0.01, k=20):
        """Find nearest matches by ratios and orientation using fixed-radius search."""

        if self.tree is None:
            return []

        query_pt = [img_ratio1, img_ratio2, img_ratio3, img_orientation]
        # Use query_ball_point (Chebyshev distance p=np.inf) to find ALL candidates 
        # within the tolerance box, ignoring arbitrary 'k' limits which fail on dense fields.
        indices = self.tree.query_ball_point(query_pt, r=tolerance, p=np.inf)

        valid_matches = []

        def score(feature_vec):
            r1, r2, r3, orient = feature_vec
            dr1 = abs(r1 - img_ratio1)
            dr2 = abs(r2 - img_ratio2)
            dr3 = abs(r3 - img_ratio3)
            dorient = 0 if orient == 0 or img_orientation == 0 else abs(orient - img_orientation)
            return dr1 + dr2 + dr3 + 2.0 * dorient

        for idx in indices:
            if idx >= len(self.triangles_data):
                continue
            hip1, hip2, hip3, orient = self.triangles_data[idx]
            feature_vec = self.tree.data[idx]
            err = score(feature_vec)
            if err <= tolerance:
                valid_matches.append({
                    'star_hips': (hip1, hip2, hip3),
                    'error': err
                })

        # Sort matches by error so best ones are first, just like k-nearest did
        valid_matches.sort(key=lambda m: m['error'])
        
        # Optional: still limit upper bound to extremely high numbers to prevent RAM flood
        return valid_matches[:200]

    def find_consensus_matches(self, triangles, tolerance=0.015, per_triangle_k=20):
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
            matches = self.find_best_match(
                t['ratio1'],
                t['ratio2'],
                t['ratio3'],
                t['orientation'],
                tolerance=tolerance,
                k=per_triangle_k
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

    def find_nearest_catalog_star(self, vector, excluded_hips=None):
        """Find nearest catalog star to a given vector using KDTree (fast O(log N))."""
        if not hasattr(self, '_catalog_vectors_cache'):
            self._build_catalog_kdtree()
        if self._catalog_kdtree is None:
            return None, 180.0
        excluded = set(excluded_hips or [])
        vec = np.asarray(vector, dtype=np.float64)
        vec = vec / np.linalg.norm(vec)
        distances, indices = self._catalog_kdtree.query(vec, k=1)
        if indices >= len(self._catalog_hips_list):
            return None, 180.0
        hip = self._catalog_hips_list[int(indices)]
        if hip in excluded:
            distances, indices = self._catalog_kdtree.query(vec, k=min(5, len(self._catalog_hips_list)))
            for dist, idx in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
                if idx < len(self._catalog_hips_list) and self._catalog_hips_list[idx] not in excluded:
                    hip = self._catalog_hips_list[int(idx)]
                    break
        catalog_vec = self._catalog_vectors_cache[hip]
        angle = float(np.arccos(np.clip(np.dot(vec, catalog_vec / np.linalg.norm(catalog_vec)), -1.0, 1.0)))
        angle_deg = np.degrees(angle)
        return hip, angle_deg

    def _build_catalog_kdtree(self):
        """Build KDTree for all catalog stars (cached, built once)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT hip_id, ra, dec FROM star_catalog ORDER BY hip_id")
        rows = cursor.fetchall()
        cursor.close()
        self._catalog_vectors_cache = {}
        self._catalog_hips_list = []
        features = []
        for hip, ra_deg, dec_deg in rows:
            vec = ra_dec_to_cartesian(float(ra_deg), float(dec_deg), radius=1.0)
            hip_int = int(hip)
            self._catalog_vectors_cache[hip_int] = vec
            self._catalog_hips_list.append(hip_int)
            features.append(vec)
        if features:
            self._catalog_kdtree = KDTree(features)
        else:
            self._catalog_kdtree = None


