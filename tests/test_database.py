import unittest

import database


class DatabaseHealthTests(unittest.TestCase):
    def test_database_health_check(self):
        database.init_database()
        self.assertTrue(database.verify_database_health())

    def test_transactional_session_rolls_back(self):
        with self.assertRaises(RuntimeError):
            with database.transactional_session():
                raise RuntimeError("force rollback")


if __name__ == "__main__":
    unittest.main()
