from abc import ABC, abstractmethod

class SlidingWindowListener(ABC):
    # Define one on_new_window method per topic:
    @abstractmethod
    def on_new_window_co_gt(self, data): pass

    @abstractmethod
    def on_new_window_pt08_s1_co(self, data): pass

    @abstractmethod
    def on_new_window_nmhc_gt(self, data): pass

    @abstractmethod
    def on_new_window_c6h6_gt(self, data): pass

    @abstractmethod
    def on_new_window_pt08_s2_nmhc(self, data): pass

    @abstractmethod
    def on_new_window_nox_gt(self, data): pass

    @abstractmethod
    def on_new_window_pt08_s3_nox(self, data): pass

    @abstractmethod
    def on_new_window_no2_gt(self, data): pass

    @abstractmethod
    def on_new_window_pt08_s4_no2(self, data): pass

    @abstractmethod
    def on_new_window_pt08_s5_o3(self, data): pass

    @abstractmethod
    def on_new_window_t(self, data): pass

    @abstractmethod
    def on_new_window_rh(self, data): pass

    @abstractmethod
    def on_new_window_ah(self, data): pass
