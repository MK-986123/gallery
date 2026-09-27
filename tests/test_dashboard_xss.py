import os
import re
import unittest

class TestDashboardXSSFix(unittest.TestCase):
    def setUp(self):
        self.file_paths = [
            'skills/built-in/mood-tracker/assets/dashboard.html',
            'Android/src/app/src/main/assets/skills/mood-tracker/assets/dashboard.html'
        ]

    def test_no_inner_html_interpolation_in_history_list(self):
        for path in self.file_paths:
            self.assertTrue(os.path.exists(path), f"File missing: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Ensure li.innerHTML is not assigned a template literal or string containing item.comment
            self.assertNotIn("li.innerHTML =", content, f"vulnerable innerHTML assignment found in {path}")
            self.assertIn("commentDiv.textContent = item.comment", content, f"safe textContent assignment missing in {path}")

    def test_files_are_identical(self):
        with open(self.file_paths[0], 'r', encoding='utf-8') as f1:
            c1 = f1.read()
        with open(self.file_paths[1], 'r', encoding='utf-8') as f2:
            c2 = f2.read()
        self.assertEqual(c1, c2, "Dashboard HTML files in skills and Android assets must match.")

if __name__ == '__main__':
    unittest.main()
