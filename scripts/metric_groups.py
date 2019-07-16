
class MetricGroup:

    def __init__(self, metrics):
        for attr in metrics:
            setattr(self, attr, 0)

class SimpleMetricGroup:

    def __init__(self, metrics):
        for attr in metrics:
            setattr(self, attr, 0)

class IterMetrics:

    metrics = ['iter_index', 'recall', 'viterbi_recall', 'logprobs', 'alpha', 'rb_score', 'zipf_p', 'zipf_L', 'zipf_alpha',
               'zipf_rules_p','zipf_rules_alpha', 'zipf_rules_L', 'ave_entropy',
               'ave_confus', 'logprob_joint_tree', 'logprob_conditional_tree', 'max_depth_by_sent_len'
               ,'surprisal', 'joint_surprisal', 'conditional_tree_surprisal', 'rule_complexity', 'accu_ave_tree_depth',
               'ave_lb_length', 'ave_rb_length', 'ave_max_lb_length', 'ave_max_rb_length', 'surprisal_variance',
               'viterbi_logprob_joint_tree', 'viterbi_joint_surprisal' , 'viterbi_ave_lb_length', 'viterbi_ave_rb_length',
               'viterbi_max_depth_by_sent_len', 'viterbi_rb_score', 'viterbi_accu_ave_tree_depth',
               'viterbi_ave_max_lb_length', 'viterbi_ave_max_rb_length', 'viterbi_rule_complexity', 'ave_span_length',
               'ave_max_span_length', 'grammar_tree_joint_logporb',
               'joint_surprisal_var', 'grammar_tree_joint_surprisal',
               'grammar_logprob', 'grammar_surprisal']

    def __init__(self, iter_index=0):
        self.iter_index = iter_index
        self.iter_metrics = [MetricGroup(self.metrics)]

    def spawn_iter(self):
        iter_index = self.iter_index + 1
        new_metric = MetricGroup(self.metrics)
        new_metric.iter_index = iter_index

        return new_metric

    @property
    def last_iter(self):
        return self.iter_metrics[-1]

    @last_iter.setter
    def last_iter(self, metric_group:MetricGroup):
        self.iter_metrics.append(metric_group)
        self.iter_index = metric_group.iter_index

class SimpleIterMetrics:

    metrics = ['iter_index', 'batch_index', 'viterbi_recall', 'logprobs', 'rb_score', 'vm_nopunc', 'vm_withpunc',
               'vas', 'sparsity', 'alpha', 'viterbi_upper', 'dev_logprobs', 'dev_vas' ]

    def __init__(self):
        self.batch_metrics = [SimpleMetricGroup(self.metrics)]
        self.best_batch_per_metric = {}
        for m in self.metrics:
            self.best_batch_per_metric[m] = self.batch_metrics[-1]
        self.burn_in_iters = 5

    def spawn_batch(self, batch_index=0, iter_index=0):
        new_metric = SimpleMetricGroup(self.metrics)
        new_metric.batch_index = batch_index
        new_metric.iter_index = iter_index
        return new_metric

    @property
    def last_batch(self):
        return self.batch_metrics[-1]

    @last_batch.setter
    def last_batch(self, metric_group:SimpleMetricGroup):
        self.batch_metrics.append(metric_group)
        self._update_best_metric_batch(metric_group)

    def _update_best_metric_batch(self, metric_group:SimpleMetricGroup):
        if metric_group.iter_index > self.burn_in_iters:
            for m in self.metrics:
                current_best = self.best_batch_per_metric[m]
                if getattr(metric_group, m) > getattr(current_best, m):
                    self.best_batch_per_metric[m] = metric_group
        else:
            return

    def write_out_last(self, fn):
        vals = []
        for name in self.metrics:
            vals.append(str(getattr(self.last_batch, name)))
        string = '\t'.join(vals)
        fn.write(string+'\n')
        fn.flush()
