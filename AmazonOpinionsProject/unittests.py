import unittest

tekst = "Good Quality Dog Food,I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most"

tekst2 ="good quality dog food,i have bought several of the vitality canned dog food products and have found them all to be of good quality. the product looks more like a stew than a processed meat and it smells better. my labrador is finicky and she appreciates this product better than  most"

class TestStringMethods(unittest.TestCase):

    def test_lower(self):
        self.assertEqual(tekst.lower(), tekst2)

    def test_islower(self):
        self.assertTrue('foo'.islower())
        self.assertFalse('Foo'.islower())

    def test_split(self):
        s = 'good quality dog food'
        self.assertEqual(s.split(), ['good', 'quality', 'dog', 'food'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

# if _name_ == '_main_':
#     unittest.main()