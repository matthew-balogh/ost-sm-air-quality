import math
from scipy.stats import norm
from global_statistics.StreamStatistics import SimpleTDigest, MovingStatistics, LossyCounting


class MKTrendDetector:
    def __init__(self, t_digest, quantile_step=5, window_size=100, rel_tol=0.01):
        self.t_digest = t_digest
        self.window_size = window_size
        self.quantile_step = quantile_step
        self.quantile_probs = [i for i in range(quantile_step, 100, quantile_step)]
        self.quantile_values = [None] * len(self.quantile_probs)
        self.S = 0
        self.Z = 0
        self.P = 0
        self.length = 0
        self.min_samples = len(self.quantile_probs)
        self.tie_counter = None
        self.rel_tol = rel_tol
        self.min_value = None
        self.max_value = None

    def _update_moving_range(self, x):
        if self.min_value is None or x < self.min_value:
            self.min_value = x
        if self.max_value is None or x > self.max_value:
            self.max_value = x
        moving_range = self.max_value - self.min_value
        eps_tie = max(moving_range * self.rel_tol, 1e-8)  # avoid zero
        if self.tie_counter is None:
            self.tie_counter = LossyCounting(epsilon=0.1, eps_tie=eps_tie)
        else:
            self.tie_counter.eps_tie = eps_tie  # update eps dynamically

    def calculate_quantiles(self, x):
        self.t_digest.update(x)
        for idx, p in enumerate(self.quantile_probs):
            self.quantile_values[idx] = self.t_digest.percentile(p)

    def approx_cdf(self, x):
        if x <= self.quantile_values[0]:
            return 0.0
        if x >= self.quantile_values[-1]:
            return 1.0
        for i in range(len(self.quantile_values)-1):
            q_i = self.quantile_values[i]
            q_next = self.quantile_values[i+1]
            p_i = self.quantile_probs[i] / 100.0
            p_next = self.quantile_probs[i+1] / 100.0
            if q_i <= x <= q_next:
                return p_i + (x - q_i)/(q_next - q_i) * (p_next - p_i)
        return 1.0

    def update(self, x):
        # reset window
        if self.length > 0 and self.length % self.window_size == 0:
            self.S = 0
            self.Z = 0
            self.P = 0
            self.length = 0
            self.tie_counter.clear()
            self.min_value = None
            self.max_value = None

        self.length += 1
        self._update_moving_range(x)
        self.calculate_quantiles(x)
        self.tie_counter.process_item(x)

        if self.length > self.min_samples:
            cdf = self.approx_cdf(x)
            self.S += self.length * (2*cdf - 1)

    def compute_variance_Z(self):
        n = self.length
        if n < 2:
            self.Z = 0
            return
        tie_groups = self.tie_counter.tie_groups()
        varS = (n*(n-1)*(2*n+5) - sum(t*(t-1)*(2*t+5) for t in tie_groups)) / 18.0
        if varS <= 0:
            self.Z = 0
            return
        if self.S > 0:
            self.Z = (self.S - 1) / math.sqrt(varS)
        elif self.S < 0:
            self.Z = (self.S + 1) / math.sqrt(varS)
        else:
            self.Z = 0

    def p_value(self):
        self.compute_variance_Z()
        self.P = 2 * (1 - norm.cdf(abs(self.Z)))

    def trend(self, alpha=0.05):
        if self.length <= self.min_samples:
            return "Not enough samples"
        self.p_value()
        if self.P < alpha:
            return "Significant increasing trend" if self.S > 0 else "Significant decreasing trend"
        return "No significant trend"



class InWindowMKTrendDetector:

    def __init__(self, verbose=False, t_digest_compression_delta=0.1, quantile_step=5, dbWriter=None):
        


        self.dbWriter = dbWriter;
        # self.co_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        # self.co_gt_trend_detector = MKTrendDetector(self.co_gt_trend_detector_t_digest, quantile_step);
        # self.co_gt_moving_stats = MovingStatistics(method="exponential", alpha=0.1);
        
        self.pt08_s1_co_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s1_co_trend_detector = MKTrendDetector(self.pt08_s1_co_trend_detector_t_digest,   quantile_step);
        self.pt08_s1_co_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        # self.nmhc_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        # self.nmhc_gt_trend_detector = MKTrendDetector(self.nmhc_gt_trend_detector_t_digest, quantile_step);
        # self.nmhc_gt_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        # self.c6h6_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        # self.c6h6_gt_trend_detector = MKTrendDetector(self.c6h6_gt_trend_detector_t_digest, quantile_step);
        # self.c6h6_gt_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        self.pt08_s2_nmhc_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s2_nmhc_trend_detector = MKTrendDetector(self.pt08_s2_nmhc_trend_detector_t_digest, quantile_step);
        self.pt08_s2_nmhc_moving_stats = MovingStatistics(method="exponential", alpha=0.1);



        # self.nox_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        # self.nox_gt_trend_detector = MKTrendDetector(self.nox_gt_trend_detector_t_digest, quantile_step);
        # self.nox_gt_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        self.pt08_s3_nox_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s3_nox_trend_detector = MKTrendDetector(self.pt08_s3_nox_trend_detector_t_digest, quantile_step);   
        self.pt08_s3_nox_moving_stats = MovingStatistics(method="exponential", alpha=0.1); 

        # self.no2_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        # self.no2_gt_trend_detector = MKTrendDetector(self.no2_gt_trend_detector_t_digest, quantile_step);
        # self.no2_gt_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        self.pt08_s4_no2_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s4_no2_trend_detector = MKTrendDetector(self.pt08_s4_no2_trend_detector_t_digest, quantile_step);
        self.pt08_s4_no2_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        self.pt08_s5_o3_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s5_o3_trend_detector = MKTrendDetector(self.pt08_s5_o3_trend_detector_t_digest, quantile_step);  
        self.pt08_s5_o3_moving_stats = MovingStatistics(method="exponential", alpha=0.1);

        self.t_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.t_trend_detector = MKTrendDetector(self.t_trend_detector_t_digest, quantile_step);
        self.t_moving_stats = MovingStatistics(method="exponential", alpha=0.1);
    
        self.ah_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.ah_trend_detector = MKTrendDetector(self.ah_trend_detector_t_digest, quantile_step);
        self.ah_moving_stats = MovingStatistics(method="exponential", alpha=0.1);
    
        self.rh_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.rh_trend_detector = MKTrendDetector(self.rh_trend_detector_t_digest, quantile_step);
        self.rh_moving_stats = MovingStatistics(method="exponential", alpha=0.1);
    
        self.verbose = verbose;
    
    
    
    # def on_new_window_co_gt(self, data):

    #     if self.verbose:
    #         print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

    #     last_item = data[-1];

    #     if last_item['value'] != -200:                
            
    #         self.co_gt_trend_detector.update(float(last_item['value']));
            
    #         self.co_gt_trend_detector.compute_variance_Z();

    #         self.co_gt_moving_stats.UpdateAll(float(last_item['value']));

    #         mean = self.co_gt_moving_stats.mean;
    #         variance = self.co_gt_moving_stats.variance;
            
    #         if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
    #             print("Writing moving stats for CO_GT: mean =", mean, ", variance =", variance);
    #             self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);
            
    #         min = self.co_gt_trend_detector.t_digest.percentile(0);
    #         q1 = self.co_gt_trend_detector.t_digest.percentile(25);
    #         median = self.co_gt_trend_detector.t_digest.percentile(50);
    #         q3 = self.co_gt_trend_detector.t_digest.percentile(75);
    #         max = self.co_gt_trend_detector.t_digest.percentile(100);
            
    #         quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
    #         if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
    #             self.dbWriter.write_quantiles({"min":min, "q1":q1, "median":median, "q3":q3, "max":max}, last_item['topic'], last_item['key']);
    #         else:
    #             print("Skipping write due to NULL quantile")

    #         trend = self.co_gt_trend_detector.trend();

    #         if trend == "Significant increasing trend":
    #             self.dbWriter.write_trend(last_item, 1, last_item['topic']);
    #         elif trend == "Significant decreasing trend":
    #             self.dbWriter.write_trend(last_item, -1, last_item['topic']);
    #         elif trend == "No significant trend":
    #             self.dbWriter.write_trend(last_item, 0, last_item['topic']);
                
    #         if self.verbose:
    #             print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_pt08_s1_co(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.pt08_s1_co_trend_detector.update(float(last_item['value']));
            self.pt08_s1_co_trend_detector.compute_variance_Z();
            self.pt08_s1_co_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.pt08_s1_co_moving_stats.mean;
            variance = self.pt08_s1_co_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for PT08_S1_CO: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.pt08_s1_co_trend_detector.t_digest.percentile(0);
            q1 = self.pt08_s1_co_trend_detector.t_digest.percentile(25);
            median = self.pt08_s1_co_trend_detector.t_digest.percentile(50);
            q3 = self.pt08_s1_co_trend_detector.t_digest.percentile(75);
            max = self.pt08_s1_co_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.pt08_s1_co_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'],self.pt08_s1_co_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'],self.pt08_s1_co_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'],self.pt08_s1_co_trend_detector.S);
            
            

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend);


    # def on_new_window_nmhc_gt(self, data):
    #     if self.verbose:
    #         print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

    #     last_item = data[-1];

    #     if last_item['value'] != -200:
    #         self.nmhc_gt_trend_detector.update(float(last_item['value']));
    #         self.nmhc_gt_trend_detector.compute_variance_Z();
    #         self.nmhc_gt_moving_stats.UpdateAll(float(last_item['value']));

    #         mean = self.nmhc_gt_moving_stats.mean;
    #         variance = self.nmhc_gt_moving_stats.variance;

    #         if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
    #             print("Writing moving stats for NMHC_GT: mean =", mean, ", variance =", variance);
    #             self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

    #         min = self.nmhc_gt_trend_detector.t_digest.percentile(0);
    #         q1 = self.nmhc_gt_trend_detector.t_digest.percentile(25);
    #         median = self.nmhc_gt_trend_detector.t_digest.percentile(50);
    #         q3 = self.nmhc_gt_trend_detector.t_digest.percentile(75);
    #         max = self.nmhc_gt_trend_detector.t_digest.percentile(100);

    #         quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
    #         if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
    #             self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
    #         else:
    #             print("Skipping write due to NULL quantile")

    #         trend = self.nmhc_gt_trend_detector.trend();

    #         if trend == "Significant increasing trend":
    #             self.dbWriter.write_trend(last_item, 1, last_item['topic']);
    #         elif trend == "Significant decreasing trend":
    #             self.dbWriter.write_trend(last_item, -1, last_item['topic']);
    #         elif trend == "No significant trend":
    #             self.dbWriter.write_trend(last_item, 0, last_item['topic']);

    #         if self.verbose:
    #             print(" + + + + + Current trend  + + + + + ", trend);


    # def on_new_window_c6h6_gt(self, data):
    #     if self.verbose:
    #         print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

    #     last_item = data[-1];

    #     if last_item['value'] != -200:
    #         self.c6h6_gt_trend_detector.update(float(last_item['value']));
    #         self.c6h6_gt_trend_detector.compute_variance_Z();
    #         self.c6h6_gt_moving_stats.UpdateAll(float(last_item['value']));

    #         mean = self.c6h6_gt_moving_stats.mean;
    #         variance = self.c6h6_gt_moving_stats.variance;

    #         if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
    #             print("Writing moving stats for C6H6_GT: mean =", mean, ", variance =", variance);
    #             self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

    #         min = self.c6h6_gt_trend_detector.t_digest.percentile(0);
    #         q1 = self.c6h6_gt_trend_detector.t_digest.percentile(25);
    #         median = self.c6h6_gt_trend_detector.t_digest.percentile(50);
    #         q3 = self.c6h6_gt_trend_detector.t_digest.percentile(75);
    #         max = self.c6h6_gt_trend_detector.t_digest.percentile(100);

    #         quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
    #         if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
    #             self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
    #         else:
    #             print("Skipping write due to NULL quantile")

    #         trend = self.c6h6_gt_trend_detector.trend();

    #         if trend == "Significant increasing trend":
    #             self.dbWriter.write_trend(last_item, 1, last_item['topic']);
    #         elif trend == "Significant decreasing trend":
    #             self.dbWriter.write_trend(last_item, -1, last_item['topic']);
    #         elif trend == "No significant trend":
    #             self.dbWriter.write_trend(last_item, 0, last_item['topic']);

    #         if self.verbose:
    #             print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_pt08_s2_nmhc(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.pt08_s2_nmhc_trend_detector.update(float(last_item['value']));
            self.pt08_s2_nmhc_trend_detector.compute_variance_Z();
            self.pt08_s2_nmhc_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.pt08_s2_nmhc_moving_stats.mean;
            variance = self.pt08_s2_nmhc_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for PT08_S2_NMHC: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.pt08_s2_nmhc_trend_detector.t_digest.percentile(0);
            q1 = self.pt08_s2_nmhc_trend_detector.t_digest.percentile(25);
            median = self.pt08_s2_nmhc_trend_detector.t_digest.percentile(50);
            q3 = self.pt08_s2_nmhc_trend_detector.t_digest.percentile(75);
            max = self.pt08_s2_nmhc_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.pt08_s2_nmhc_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.pt08_s2_nmhc_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.pt08_s2_nmhc_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.pt08_s2_nmhc_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend);


    # def on_new_window_nox_gt(self, data):
    #     if self.verbose:
    #         print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

    #     last_item = data[-1];

    #     if last_item['value'] != -200:
    #         self.nox_gt_trend_detector.update(float(last_item['value']));
    #         self.nox_gt_trend_detector.compute_variance_Z();
    #         self.nox_gt_moving_stats.UpdateAll(float(last_item['value']));

    #         mean = self.nox_gt_moving_stats.mean;
    #         variance = self.nox_gt_moving_stats.variance;

    #         if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
    #             print("Writing moving stats for NOX_GT: mean =", mean, ", variance =", variance);
    #             self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

    #         min = self.nox_gt_trend_detector.t_digest.percentile(0);
    #         q1 = self.nox_gt_trend_detector.t_digest.percentile(25);
    #         median = self.nox_gt_trend_detector.t_digest.percentile(50);
    #         q3 = self.nox_gt_trend_detector.t_digest.percentile(75);
    #         max = self.nox_gt_trend_detector.t_digest.percentile(100);

    #         quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
    #         if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
    #             self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
    #         else:
    #             print("Skipping write due to NULL quantile")

    #         trend = self.nox_gt_trend_detector.trend();

    #         if trend == "Significant increasing trend":
    #             self.dbWriter.write_trend(last_item, 1, last_item['topic']);
    #         elif trend == "Significant decreasing trend":
    #             self.dbWriter.write_trend(last_item, -1, last_item['topic']);
    #         elif trend == "No significant trend":
    #             self.dbWriter.write_trend(last_item, 0, last_item['topic']);

    #         if self.verbose:
    #             print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_pt08_s3_nox(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.pt08_s3_nox_trend_detector.update(float(last_item['value']));
            self.pt08_s3_nox_trend_detector.compute_variance_Z();
            self.pt08_s3_nox_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.pt08_s3_nox_moving_stats.mean;
            variance = self.pt08_s3_nox_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for PT08_S3_NOX: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.pt08_s3_nox_trend_detector.t_digest.percentile(0);
            q1 = self.pt08_s3_nox_trend_detector.t_digest.percentile(25);
            median = self.pt08_s3_nox_trend_detector.t_digest.percentile(50);
            q3 = self.pt08_s3_nox_trend_detector.t_digest.percentile(75);
            max = self.pt08_s3_nox_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.pt08_s3_nox_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.pt08_s3_nox_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.pt08_s3_nox_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.pt08_s3_nox_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend);


    # def on_new_window_no2_gt(self, data):
    #     if self.verbose:
    #         print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

    #     last_item = data[-1];

    #     if last_item['value'] != -200:
    #         self.no2_gt_trend_detector.update(float(last_item['value']));
    #         self.no2_gt_trend_detector.compute_variance_Z();
    #         self.no2_gt_moving_stats.UpdateAll(float(last_item['value']));

    #         mean = self.no2_gt_moving_stats.mean;
    #         variance = self.no2_gt_moving_stats.variance;

    #         if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
    #             print("Writing moving stats for NO2_GT: mean =", mean, ", variance =", variance);
    #             self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

    #         min = self.no2_gt_trend_detector.t_digest.percentile(0);
    #         q1 = self.no2_gt_trend_detector.t_digest.percentile(25);
    #         median = self.no2_gt_trend_detector.t_digest.percentile(50);
    #         q3 = self.no2_gt_trend_detector.t_digest.percentile(75);
    #         max = self.no2_gt_trend_detector.t_digest.percentile(100);

    #         quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
    #         if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
    #             self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
    #         else:
    #             print("Skipping write due to NULL quantile")

    #         trend = self.no2_gt_trend_detector.trend();

    #         if trend == "Significant increasing trend":
    #             self.dbWriter.write_trend(last_item, 1, last_item['topic']);
    #         elif trend == "Significant decreasing trend":
    #             self.dbWriter.write_trend(last_item, -1, last_item['topic']);
    #         elif trend == "No significant trend":
    #             self.dbWriter.write_trend(last_item, 0, last_item['topic']);

    #         if self.verbose:
    #             print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_pt08_s4_no2(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.pt08_s4_no2_trend_detector.update(float(last_item['value']));
            self.pt08_s4_no2_trend_detector.compute_variance_Z();
            self.pt08_s4_no2_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.pt08_s4_no2_moving_stats.mean;
            variance = self.pt08_s4_no2_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for PT08_S4_NO2: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.pt08_s4_no2_trend_detector.t_digest.percentile(0);
            q1 = self.pt08_s4_no2_trend_detector.t_digest.percentile(25);
            median = self.pt08_s4_no2_trend_detector.t_digest.percentile(50);
            q3 = self.pt08_s4_no2_trend_detector.t_digest.percentile(75);
            max = self.pt08_s4_no2_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.pt08_s4_no2_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.pt08_s4_no2_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.pt08_s4_no2_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.pt08_s4_no2_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_pt08_s5_o3(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.pt08_s5_o3_trend_detector.update(float(last_item['value']));
            self.pt08_s5_o3_trend_detector.compute_variance_Z();
            self.pt08_s5_o3_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.pt08_s5_o3_moving_stats.mean;
            variance = self.pt08_s5_o3_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for PT08_S5_O3: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.pt08_s5_o3_trend_detector.t_digest.percentile(0);
            q1 = self.pt08_s5_o3_trend_detector.t_digest.percentile(25);
            median = self.pt08_s5_o3_trend_detector.t_digest.percentile(50);
            q3 = self.pt08_s5_o3_trend_detector.t_digest.percentile(75);
            max = self.pt08_s5_o3_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.pt08_s5_o3_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.pt08_s5_o3_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.pt08_s5_o3_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.pt08_s5_o3_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_t(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.t_trend_detector.update(float(last_item['value']));
            self.t_trend_detector.compute_variance_Z();
            self.t_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.t_moving_stats.mean;
            variance = self.t_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for T: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.t_trend_detector.t_digest.percentile(0);
            q1 = self.t_trend_detector.t_digest.percentile(25);
            median = self.t_trend_detector.t_digest.percentile(50);
            q3 = self.t_trend_detector.t_digest.percentile(75);
            max = self.t_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.t_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.t_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.t_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.t_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend);


    def on_new_window_ah(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.ah_trend_detector.update(float(last_item['value']));
            self.ah_trend_detector.compute_variance_Z();
            self.ah_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.ah_moving_stats.mean;
            variance = self.ah_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for AH: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.ah_trend_detector.t_digest.percentile(0);
            q1 = self.ah_trend_detector.t_digest.percentile(25);
            median = self.ah_trend_detector.t_digest.percentile(50);
            q3 = self.ah_trend_detector.t_digest.percentile(75);
            max = self.ah_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.ah_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.ah_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.ah_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.ah_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend)
                print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_rh(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1];

        if last_item['value'] != -200:
            self.rh_trend_detector.update(float(last_item['value']));
            self.rh_trend_detector.compute_variance_Z();
            self.rh_moving_stats.UpdateAll(float(last_item['value']));

            mean = self.rh_moving_stats.mean;
            variance = self.rh_moving_stats.variance;

            if mean is not None and variance is not None and not (isinstance(mean, float) and math.isnan(mean)) and not (isinstance(variance, float) and math.isnan(variance)):
                print("Writing moving stats for RH: mean =", mean, ", variance =", variance);
                self.dbWriter.write_moving_statistics({"mean":mean, "variance":variance}, last_item['topic'], last_item['key']);

            min = self.rh_trend_detector.t_digest.percentile(0);
            q1 = self.rh_trend_detector.t_digest.percentile(25);
            median = self.rh_trend_detector.t_digest.percentile(50);
            q3 = self.rh_trend_detector.t_digest.percentile(75);
            max = self.rh_trend_detector.t_digest.percentile(100);

            quantiles = {"min":min, "q1":q1, "median":median, "q3":q3, "max":max}
            if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in quantiles.values()):
                self.dbWriter.write_quantiles(quantiles, last_item['topic'], last_item['key']);
            else:
                print("Skipping write due to NULL quantile")

            trend = self.rh_trend_detector.trend();

            if trend == "Significant increasing trend":
                self.dbWriter.write_trend(last_item, 1, last_item['topic'], self.rh_trend_detector.S);
            elif trend == "Significant decreasing trend":
                self.dbWriter.write_trend(last_item, -1, last_item['topic'], self.rh_trend_detector.S);
            else :
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.rh_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend)
                print(" + + + + + TOPIC  + + + + + ", last_item['topic'])
