import math
from scipy.stats import norm
from global_statistics.StreamStatistics import SimpleTDigest


class MKTrendDetector:

    def __init__(self,t_digest, quantile_step = 5):
        self.t_digest = t_digest;
        self.quantile_probs = [i/100 for i in range(quantile_step, 100, quantile_step)]  # 0.05, 0.10, ..., 0.95
        self.quantile_values = [None]*len(self.quantile_probs) 
        self.length = 0;
        self.S = 0;
        self.Z = 0;
        self.P = 0;
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
        if x >= self.quantile_probs[-1]:
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
        
                
        if self.length <= self.min_samples:
            print("Not enough samples for Trend detection");
        # Step 2: compute new St by adding the sample contribution to St-1. 
        else:
            if self.length > 0: 
                cdf = self.approx_cdf(x);
                self.S += self.length * (2*cdf - 1)

        # Step 3: update count
        self.length += 1;                

            
        
    def compute_variance_Z(self):
        # Let's assume no ties.
        varS = self.length * (self.length-1) * (2*self.length + 5) / 18
        if self.S > 0:
            self.Z = (self.S - 1) / math.sqrt(varS)
        elif self.S < 0:
            self.Z = (self.S + 1) / math.sqrt(varS)
        else:
            self.Z = 0.0;




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
    def __init__(self, verbose=False, t_digest_compression_delta=0.1, quantile_step=5):
        self.co_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.co_gt_trend_detector = MKTrendDetector(self.co_gt_trend_detector_t_digest, quantile_step);
        
        self.pt08_s1_co_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s1_co_trend_detector = MKTrendDetector(self.pt08_s1_co_trend_detector_t_digest,   quantile_step);

        self.nmhc_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.nmhc_gt_trend_detector = MKTrendDetector(self.nmhc_gt_trend_detector_t_digest, quantile_step);

        self.c6h6_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.c6h6_gt_trend_detector = MKTrendDetector(self.c6h6_gt_trend_detector_t_digest, quantile_step);

        self.pt08_s2_nmhc_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s2_nmhc_trend_detector = MKTrendDetector(self.pt08_s2_nmhc_trend_detector_t_digest, quantile_step);

        self.nox_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.nox_gt_trend_detector = MKTrendDetector(self.nox_gt_trend_detector_t_digest, quantile_step);

        self.pt08_s3_nox_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s3_nox_trend_detector = MKTrendDetector(self.pt08_s3_nox_trend_detector_t_digest, quantile_step);    

        self.no2_gt_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.no2_gt_trend_detector = MKTrendDetector(self.no2_gt_trend_detector_t_digest, quantile_step);

        self.pt08_s4_no2_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s4_no2_trend_detector = MKTrendDetector(self.pt08_s4_no2_trend_detector_t_digest, quantile_step);

        self.pt08_s5_o3_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.pt08_s5_o3_trend_detector = MKTrendDetector(self.pt08_s5_o3_trend_detector_t_digest, quantile_step);  

        self.t_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.t_trend_detector = MKTrendDetector(self.t_trend_detector_t_digest, quantile_step);
    
        self.ah_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.ah_trend_detector = MKTrendDetector(self.ah_trend_detector_t_digest, quantile_step);
    
        self.rh_trend_detector_t_digest = SimpleTDigest(t_digest_compression_delta);
        self.rh_trend_detector = MKTrendDetector(self.rh_trend_detector_t_digest, quantile_step);
    
        self.verbose = verbose;
    
        

    
    def on_new_window_co_gt(self, data):

        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");


        last_item = data[-1];
        
        self.co_gt_trend_detector.update(float(last_item['value']));
        
        self.co_gt_trend_detector.compute_variance_Z();
        
        # trend = self.trend();

        trend = self.co_gt_trend_detector.trend();

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend);
        
        
        

        # self.write_data("environment",{"topic":"co_gt"},{"value":last_item['value']},last_item['key'])




    def on_new_window_pt08_s1_co(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.pt08_s1_co_trend_detector.update(float(last_item['value']))
        
        self.pt08_s1_co_trend_detector.compute_variance_Z()
        
        trend = self.pt08_s1_co_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)


    def on_new_window_nmhc_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.nmhc_gt_trend_detector.update(float(last_item['value']))
        
        self.nmhc_gt_trend_detector.compute_variance_Z()
        
        trend = self.nmhc_gt_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)


    def on_new_window_c6h6_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.c6h6_gt_trend_detector.update(float(last_item['value']))
        
        self.c6h6_gt_trend_detector.compute_variance_Z()
        
        trend = self.c6h6_gt_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)


    def on_new_window_pt08_s2_nmhc(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.pt08_s2_nmhc_trend_detector.update(float(last_item['value']))
        
        self.pt08_s2_nmhc_trend_detector.compute_variance_Z()
        
        trend = self.pt08_s2_nmhc_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)


    def on_new_window_nox_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.nox_gt_trend_detector.update(float(last_item['value']))
        
        self.nox_gt_trend_detector.compute_variance_Z()
        
        trend = self.nox_gt_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)


    def on_new_window_pt08_s3_nox(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.pt08_s3_nox_trend_detector.update(float(last_item['value']))
        
        self.pt08_s3_nox_trend_detector.compute_variance_Z()
        
        trend = self.pt08_s3_nox_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_no2_gt(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.no2_gt_trend_detector.update(float(last_item['value']))
        
        self.no2_gt_trend_detector.compute_variance_Z()
        
        trend = self.no2_gt_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_pt08_s4_no2(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.pt08_s4_no2_trend_detector.update(float(last_item['value']))
        
        self.pt08_s4_no2_trend_detector.compute_variance_Z()
        
        trend = self.pt08_s4_no2_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_pt08_s5_o3(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.pt08_s5_o3_trend_detector.update(float(last_item['value']))
        
        self.pt08_s5_o3_trend_detector.compute_variance_Z()
        
        trend = self.pt08_s5_o3_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_t(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.t_trend_detector.update(float(last_item['value']))
        
        self.t_trend_detector.compute_variance_Z()
        
        trend = self.t_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_ah(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.ah_trend_detector.update(float(last_item['value']))
        
        self.ah_trend_detector.compute_variance_Z()
        
        trend = self.ah_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])


    def on_new_window_rh(self, data):
        if self.verbose:
            print("++++++++++++++++++++++++++++++++++++++ TREND detector +++++++++++++++++++++++++++++++++");

        last_item = data[-1]
        
        self.rh_trend_detector.update(float(last_item['value']))
        
        self.rh_trend_detector.compute_variance_Z()
        
        trend = self.rh_trend_detector.trend()

        if self.verbose:
            print(" + + + + + Current trend  + + + + + ", trend)
            print(" + + + + + TOPIC  + + + + + ", last_item['topic'])