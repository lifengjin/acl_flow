class MetricGroup:

    iter_index : int = 0
    recall : float = 0
    logprobs : float = 0.
    alpha : float = 0.
    rb_score : float = 0.
    zipf_p : float = 0.
    zipf_alpha : float = 0.
    zipf_L : float = 0.
    zipf_rules_p : float = 0.
    zipf_rules_alpha : float = 0.
    zipf_rules_L : float = 0.
    ave_entropy : float = 0.
    ave_confus : float = 0.
    logprob_joint_tree : float = 0.
    logprob_conditional_tree : float = 0.
    max_depth_by_sent_len : float = 0.
    surprisal : float = 0.
    joint_surprisal : float = 0.
    conditional_tree_surprisal : float = 0.
    rule_complexity : float = 0.
    accu_ave_tree_depth : float = 0.
    ave_lb_length : float = 0.
    ave_rb_length : float = 0.
    ave_max_lb_length : float = 0.
    ave_max_rb_length : float = 0.

    surprisal_variance : float = 0.
    viterbi_recall : float = 0.
    viterbi_logprob_joint_tree : float = 0.
    viterbi_joint_surprisal : float = 0.
    viterbi_ave_lb_length : float = 0.
    viterbi_ave_rb_length : float = 0.
    viterbi_max_depth_by_sent_len : float = 0.
    viterbi_rb_score : float = 0.
    ###

    viterbi_accu_ave_tree_depth : float = 0.
    viterbi_ave_max_lb_length : float = 0.
    viterbi_ave_max_rb_length : float = 0.
    viterbi_rule_complexity : float = 0.

    ###
    ave_max_span_length : float = 0.
    ave_span_length : float = 0.
    joint_surprisal_var : float = 0.
    grammar_tree_joint_logporb : float = 0.
    grammar_tree_joint_surprisal : float = 0.
    grammar_logprob : float = 0.
    grammar_surprisal : float = 0.


class SimpleMetricGroup:
    batch_index : int = 0
    iter_index : int = 0
    viterbi_recall : float = 0.
    logprobs : float = -float('inf')
    alpha : float = 0.
    rb_score : float = 0.
    vm_nopunc : float = 0.
    vm_withpunc : float = 0.
    vas : float = 0.
    sparsity : float = 0.
    viterbi_upper : float = -float('inf')
    dev_logprobs : float = 0.
    dev_vas : float = 0.


class IterMetrics:
    iter_index : int
    iter_metrics : list
    last_iter : MetricGroup
    metrics : list
    def spawn_iter(self) -> MetricGroup :
        ...

class SimpleIterMetrics:
    iter_index : int
    batch_index : int
    batch_metrics : list
    last_batch : SimpleMetricGroup
    metrics : list
    def spawn_batch(self) -> SimpleMetricGroup :
        ...
    def write_out_last(self, fn):
        ...