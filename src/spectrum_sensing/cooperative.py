"""Cooperative sensing: K nodes report 1-bit decisions, OR rule at the fusion center."""


def or_fusion(decisions):
    """OR rule: declare busy if any node declares busy."""
    return any(decisions)


def cooperative_pd(pd_single, k):
    """Pd of the OR rule with K independent nodes, each with detection prob pd_single."""
    return 1.0 - (1.0 - pd_single) ** k


def cooperative_pfa(pfa_single, k):
    """Pfa of the OR rule with K independent nodes, each with false-alarm prob pfa_single."""
    return 1.0 - (1.0 - pfa_single) ** k
