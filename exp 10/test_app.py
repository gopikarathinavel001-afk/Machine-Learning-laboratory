import json
import unittest
from app import app, get_history, DB_PATH
import os

class TestCustomerPurchaseApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page_loads(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Customer Purchase Prediction", html)
        self.assertIn("Step 1", html)
        self.assertIn("Step 2", html)
        self.assertIn("Step 3", html)
        self.assertIn("Step 4", html)
        self.assertIn("Step 5", html)
        self.assertIn("Step 6", html)
        self.assertIn("Step 7", html)
        self.assertIn("Preprocessing Completed Successfully", html)
        self.assertIn("Model Training Completed Successfully", html)
        self.assertIn("Accuracy", html)
        self.assertIn("Confusion Matrix", html)

    def test_prediction_api_yes(self):
        payload = {
            'age': 28,
            'previous_purchases': 6,
            'website_time': 24.5,
            'products_viewed': 14,
            'previous_spending': 5200,
            'discount': 25
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['result']['prediction'], 'YES')
        self.assertGreater(data['result']['probability'], 50.0)

    def test_prediction_api_no(self):
        payload = {
            'age': 45,
            'previous_purchases': 0,
            'website_time': 2.0,
            'products_viewed': 1,
            'previous_spending': 0,
            'discount': 0
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['result']['prediction'], 'NO')
        self.assertLess(data['result']['probability'], 50.0)

    def test_validation_errors(self):
        # Negative age
        payload = {
            'age': -5,
            'previous_purchases': 2,
            'website_time': 10,
            'products_viewed': 5,
            'previous_spending': 100,
            'discount': 10
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn("Age must be a valid positive value", data['error'])

        # Discount > 100
        payload['age'] = 25
        payload['discount'] = 150
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("Discount offered must be between 0% and 100%", data['error'])

    def test_history_logging_and_clear(self):
        # Trigger a prediction first
        payload = {
            'age': 30,
            'previous_purchases': 4,
            'website_time': 15.0,
            'products_viewed': 9,
            'previous_spending': 3000,
            'discount': 15
        }
        res = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)

        history = get_history()
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)

if __name__ == '__main__':
    unittest.main()
