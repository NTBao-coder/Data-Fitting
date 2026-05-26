
import unittest
import numpy as np
from sklearn.linear_model import Ridge as SklearnRidge, Lasso as SklearnLasso

try:
    from part1.ridge_lasso import MyRidge, MyLasso
except ImportError:
    class MyRidge:
        def __init__(self, alpha): self.alpha = alpha
        def fit(self, X, y): self.coef_ = np.zeros(X.shape[1]); self.intercept_ = 0.0
    class MyLasso:
        def __init__(self, alpha): self.alpha = alpha
        def fit(self, X, y): self.coef_ = np.zeros(X.shape[1]); self.intercept_ = 0.0

class TestRidgeLassoImplementation(unittest.TestCase):
    
    def setUp(self):
        """Khởi tạo dữ liệu mẫu cố định để test thuật toán"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5) 
        true_w = np.array([1.5, -2.0, 0.0, 3.4, 0.0]) 
        self.y = self.X @ true_w + 1.2 + np.random.randn(100) * 0.1
        self.alpha = 0.5 

    def test_ridge_coefficients_match_sklearn(self):
        """[Chuẩn] Đối chiếu hệ số Ridge với Sklearn"""
        sk_model = SklearnRidge(alpha=self.alpha, fit_intercept=True)
        sk_model.fit(self.X, self.y)
        
        my_model = MyRidge(alpha=self.alpha)
        my_model.fit(self.X, self.y)
        
        np.testing.assert_allclose(my_model.coef_, sk_model.coef_, atol=1e-3,
                                    err_msg="LỖI: Hệ số Ridge không khớp Sklearn ở điều kiện chuẩn!")
        self.assertAlmostEqual(my_model.intercept_, sk_model.intercept_, places=3)
        print("✅ Test: Ridge hoạt động chính xác ở điều kiện chuẩn.")

    def test_lasso_coefficients_match_sklearn(self):
        """[Chuẩn] Đối chiếu giải thuật Lasso (Coordinate Descent) với Sklearn"""
        sk_model = SklearnLasso(alpha=self.alpha, fit_intercept=True, max_iter=3000)
        sk_model.fit(self.X, self.y)
        
        my_model = MyLasso(alpha=self.alpha)
        my_model.fit(self.X, self.y)
        
        np.testing.assert_allclose(my_model.coef_, sk_model.coef_, atol=1e-2,
                                    err_msg="LỖI: Hệ số Lasso không khớp Sklearn ở điều kiện chuẩn!")
        self.assertAlmostEqual(my_model.intercept_, sk_model.intercept_, places=2)
        print("✅ Test: Lasso hoạt động chính xác ở điều kiện chuẩn.")

    def test_lasso_sparsity_property(self):
        """[Đặc trưng] Kiểm tra tính thưa hóa (Sparsity) ép trọng số rác về ĐÚNG 0"""
        my_model = MyLasso(alpha=1.0) 
        my_model.fit(self.X, self.y)
        
        # Đặc trưng tại index 2 và 4 là nhiễu, hệ số góc bắt buộc phải bằng 0 hoàn toàn
        self.assertEqual(my_model.coef_[2], 0.0, msg="LỖI: Lasso không ép được biến nhiễu về đúng 0.0!")
        self.assertEqual(my_model.coef_[4], 0.0, msg="LỖI: Lasso không ép được biến nhiễu về đúng 0.0!")
        print("✅ Test: Lasso triệt tiêu biến nhiễu về 0 hoàn toàn đạt Sparsity.")

    def test_alpha_zero_boundary(self):
        """[Edge Case] Khi alpha = 0, mô hình phải tương đương với nghiệm OLS giải tích"""
        # Khi alpha = 0, Ridge chính là nghiệm OLS: beta = (X^T X)^(-1) X^T y
        sk_ridge = SklearnRidge(alpha=0.0, fit_intercept=True)
        sk_ridge.fit(self.X, self.y)
        
        my_ridge = MyRidge(alpha=0.0)
        my_ridge.fit(self.X, self.y)
        
        np.testing.assert_allclose(my_ridge.coef_, sk_ridge.coef_, atol=1e-3,
                                    err_msg="LỖI: Khi alpha=0, Ridge không hội tụ về nghiệm OLS!")
        print("✅ Test: Biên alpha = 0 (OLS nghiệm giải tích) vượt qua thành công.")

    def test_extreme_high_alpha_lasso(self):
        """[Edge Case] Khi alpha cực lớn, Lasso phải ép TOÀN BỘ hệ số góc về 0"""
        my_model = MyLasso(alpha=100.0) 
        my_model.fit(self.X, self.y)
        
        # Toàn bộ vector coefficients phải là mảng zero
        np.testing.assert_array_equal(my_model.coef_, np.zeros(self.X.shape[1]),
                                      err_msg="LỖI: Alpha cực lớn nhưng Lasso không ép hết các biến về 0!")
        print("✅ Test: Biên alpha cực đại, toàn bộ trọng số Lasso bị triệt tiêu hoàn toàn.")

    def test_y_shape_resilience(self):
        """[Robustness] Kiểm tra tính bền vững khi mảng mục tiêu y bị đổi dạng cột (n, 1)"""
        y_column = self.y.reshape(-1, 1) # Chuyển thành mảng 2D dạng cột
        
        my_ridge = MyRidge(alpha=self.alpha)
        # Nếu hàm fit không xử lý ép mảng phẳng .flatten() hoặc .ravel(), nó sẽ bị lỗi ma trận ở đây
        try:
            my_ridge.fit(self.X, y_column)
            self.assertEqual(my_ridge.coef_.ndim, 1, msg="LỖI: Output coef_ phải luôn là mảng 1 chiều!")
            print("✅ Test: Mô hình xử lý tốt cấu trúc mảng đầu vào y dạng (n, 1).")
        except Exception as e:
            self.fail(f"LỖI: Thuật toán bị crash khi y là mảng cột 2D! Chi tiết: {e}")

if __name__ == "__main__":
    unittest.main()