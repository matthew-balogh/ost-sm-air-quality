import math
from scipy.stats import norm
from global_statistics.StreamStatistics import SimpleTDigest, MovingStatistics, LossyCounting


class MKTrendDetector:

    def __init__(self,t_digest, quantile_step = 5):
        self.t_digest = t_digest;
        self.quantile_probs = [i for i in range(quantile_step, 100, quantile_step)]  # 0.05, 0.10, ..., 0.95
        self.quantile_values = [None]*len(self.quantile_probs) 
        self.length = 0;
        self.S = 0;
        self.Z = 0;
        self.P = 0;
        self.count_approx =  LossyCounting(epsilon=0.01);
        self.min_samples = len(self.quantile_probs);
    


    def calculate_quantiles_tdigest(self,sample):
        # First we should get a decent number of samples
        self.length+= 1;

        self.t_digest.update(sample);
        
        # if self.length <= self.min_samples:
        #     print("Not enough samples for Trend detection");
        # else:
        for idx, p in enumerate(self.quantile_probs):
            self.quantile_values[idx] = self.t_digest.percentile(p);
    


    def approx_cdf(self, x):     
        '''
        Approximate the CDF function, and interpolate. <=>  calculate P(X<= xi) = F(xi)
        '''     

        ## Extreme cases
        if x <= self.quantile_values[0]:
            return 0.0
        if x >= self.quantile_values[-1]:
            return 1.0
        
        for i in range(len(self.quantile_values)-1):
            q_i = self.quantile_values[i]
            q_next = self.quantile_values[i+1]
            p_i = self.quantile_probs[i]
            p_next = self.quantile_probs[i+1]

            if q_i <= x <= q_next:
                return p_i + (x - q_i)/(q_next - q_i) * (p_next - p_i)
            
        return 1.0
    


    def update(self, x):
        """
        Update the online MK statistic with a new value x
        """
        # Step 1: update quantiles (may use small history initially)
        self.calculate_quantiles_tdigest(x);
        self.count_approx.process_item(x);
        
        # Step 2: compute new St by adding the sample contribution to St-1
        if self.length <= self.min_samples:
            print("Not enough samples for Trend detection")
        else:
            if self.length > 0: 
                cdf = self.approx_cdf(x)
                self.S += self.length * (2*cdf - 1)
    
        # Step 3: update count (only once)
        # self.length is already incremented in calculate_quantiles_tdigest
            
        
    def compute_variance_Z(self):
        # Let's assume no ties.
        if self.count_approx is not None:
            ties = self.count_approx.estimated_counts();
            print("============= Ties ===============", ties); 
        tie_groups = [tj for tj  in ties.values() if tj > 1];
        print("============= Tie groups ===============", tie_groups);
        varS = (self.length*(self.length-1)*(2*self.length+5)
            - sum(tj*(tj-1)*(2*tj+5) for tj in tie_groups)) / 18

        if self.S > 0:
            self.Z = (self.S - 1) / math.sqrt(varS)
        elif self.S < 0:
            self.Z = (self.S + 1) / math.sqrt(varS)
        else:
            self.Z = 0.0



    def p_value(self):
        self.compute_variance_Z()
        self.P = 2 * (1 - norm.cdf(abs(self.Z)))  # two-sided



    def trend(self, alpha=0.05):        
        self.p_value();
        if self.P < alpha:
            if self.S > 0:
                return "Significant increasing trend"
            else:
                return "Significant decreasing trend"
        else:
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
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
            elif trend == "No significant trend":
                self.dbWriter.write_trend(last_item, 0, last_item['topic'], self.rh_trend_detector.S);

            if self.verbose:
                print(" + + + + + Current trend  + + + + + ", trend)
                print(" + + + + + TOPIC  + + + + + ", last_item['topic'])
