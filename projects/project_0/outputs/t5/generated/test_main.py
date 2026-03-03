import unittest
from core_modules import Database, User

class TestDatabaseIntegration(unittest.TestCase):
    def setUp(self):
        self.db = Database()

    def test_user_creation_and_retrieval(self):
        user = User("John", "Doe")
        user_id = self.db.save(user)

        retrieved_user = self.db.get_user_by_id(user_id)
        self.assertEqual(retrieved_user.name, "John")

if __name__ == '__main__':
    unittest.main()