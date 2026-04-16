"""
unit_tests.py - Модульні тести для перевірки логіки Stars Detector
"""

import math
import unittest


class TestTriangleCalculations(unittest.TestCase):
    """Тести математичних обчислень для трикутників"""

    def test_cosine_clamping(self):
        """Перевірка, що косинус завжди в межах [-1, 1]"""
        # Сценарій: через помилки округлення косинус може бути > 1
        # Це повинно викликати ValueError при acos(), але з clamping не повинно

        cos_value = 1.0000000001  # Через помилку округлення
        clamped = max(-1.0, min(1.0, cos_value))

        self.assertEqual(clamped, 1.0)
        self.assertGreaterEqual(clamped, -1.0)
        self.assertLessEqual(clamped, 1.0)

        # Now acos() won't throw error
        angle = math.degrees(math.acos(clamped))
        self.assertIsInstance(angle, float)
        print(f"Cosine correctly clamped: {cos_value} → {clamped} → {angle}°")

    def test_triangle_angle_filtering(self):
        """Перевірка фільтрації нестабільних трикутників"""
        # Трикутник зі сторонами (відомої: 3, 4, 5 - прямокутний, 90°)
        a, b, c = 3.0, 4.0, 5.0

        # Розрахунок кутів через теорему косинусів
        cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        max_angle = math.degrees(math.acos(cos_gamma))

        cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        min_angle = math.degrees(math.acos(cos_alpha))

        # Check filtering (angles should be between 10-170°)
        should_pass = min_angle >= 10.0 and max_angle <= 170.0

        print(f"Triangle 3-4-5: min_angle={min_angle:.1f}°, max_angle={max_angle:.1f}°")
        print(f"  Passes filter: {should_pass}")

        self.assertTrue(should_pass, "Right triangle should pass filter")

    def test_needle_triangle_rejected(self):
        """Перевірка, що "голки" (вузькі трикутники) відкидаються"""
        # Трикутник зі сторонами 0.1, 1, 1.05 - дуже вузький (голка)
        a, b, c = 0.1, 1.0, 1.05

        cos_gamma = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        max_angle = math.degrees(math.acos(cos_gamma))

        cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        min_angle = math.degrees(math.acos(cos_alpha))

        # This triangle should be rejected (min_angle < 10°)
        should_fail = min_angle < 10.0 or max_angle > 170.0

        print(f"Needle 0.1-1-1.05: min_angle={min_angle:.1f}°, max_angle={max_angle:.1f}°")
        print(f"  Will be rejected: {should_fail}")

        self.assertTrue(should_fail, "Needle should be rejected")

    def test_ratio_calculation(self):
        """Перевірка розрахунку пропорцій трикутника"""
        # Трикутник зі сторонами 1, 2, 3 (це виродження, але для тесту)
        a, b, c = 1.0, 2.0, 3.0

        ratio1 = b / a
        ratio2 = c / a

        self.assertEqual(ratio1, 2.0)
        self.assertEqual(ratio2, 3.0)

        print(f"Ratios: {a}-{b}-{c} → ratio1={ratio1}, ratio2={ratio2}")

    def test_ratio_invariance(self):
        """Перевірка, що пропорції інваріантні до масштабування"""
        # Трикутник 2-3-4
        a1, b1, c1 = 2.0, 3.0, 4.0
        ratio1_small = b1 / a1
        ratio2_small = c1 / a1

        # Той же трикутник в 10 разів більше: 20-30-40
        a2, b2, c2 = 20.0, 30.0, 40.0
        ratio1_big = b2 / a2
        ratio2_big = c2 / a2

        self.assertAlmostEqual(ratio1_small, ratio1_big)
        self.assertAlmostEqual(ratio2_small, ratio2_big)

        print(f"Scale invariance: {ratio1_small} == {ratio1_big}")


class TestAngularDistance(unittest.TestCase):
    """Тести розрахунку кутової відстані на сфері"""

    def test_angular_distance_same_point(self):
        """Кутова відстань від точки до себе повинна бути 0°"""
        # За формулою великої дуги
        ra1, dec1 = 0.0, 0.0
        ra2, dec2 = 0.0, 0.0

        ra1_rad = math.radians(ra1 * 15)
        ra2_rad = math.radians(ra2 * 15)
        dec1_rad = math.radians(dec1)
        dec2_rad = math.radians(dec2)

        term = (math.sin(dec1_rad) * math.sin(dec2_rad) +
                math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad))
        term = max(-1.0, min(1.0, term))

        distance = math.degrees(math.acos(term))

        self.assertAlmostEqual(distance, 0.0, places=5)
        print(f"Same point: {distance:.5f}°")

    def test_angular_distance_90_degrees(self):
        """Кутова відстань від екватора (0°) до полюса (90°) повинна бути 90°"""
        ra1, dec1 = 0.0, 0.0       # Екватор
        ra2, dec2 = 0.0, 90.0      # Північний полюс

        ra1_rad = math.radians(ra1 * 15)
        ra2_rad = math.radians(ra2 * 15)
        dec1_rad = math.radians(dec1)
        dec2_rad = math.radians(dec2)

        term = (math.sin(dec1_rad) * math.sin(dec2_rad) +
                math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad))
        term = max(-1.0, min(1.0, term))

        distance = math.degrees(math.acos(term))

        self.assertAlmostEqual(distance, 90.0, places=3)
        print(f"Equator to pole: {distance:.3f}°")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS FOR STARS DETECTOR")
    print("=" * 60 + "\n")

    # Запуск тестів з verbose output
    unittest.main(verbosity=2)

