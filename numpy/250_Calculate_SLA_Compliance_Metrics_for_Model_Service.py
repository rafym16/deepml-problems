def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.

    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance

    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if requests == [] or len(requests) == 0:
        return {}

    success_sla = [req for req in requests if req['status'] == 'success']
    meet_sla = [req for req in requests if req['status'] == 'success' and req['latency_ms'] < latency_sla_ms]
    error_sla = [req for req in requests if req['status'] != 'success']

    try:
        meet_rate = len(meet_sla) / len(success_sla) * 100
    except ZeroDivisionError:
        meet_rate = 0.0
    try:
        error_rate = len(error_sla) / len(requests) * 100
    except ZeroDivisionError:
        error_rate = 0.0
    try:
        overall_rate = len(meet_sla) / len(requests) * 100
    except ZeroDivisionError:
        overall_rate = 0.0

    overall_sla_compliance = {
        'latency_sla_compliance': round(meet_rate, 2),
        'error_rate': round(error_rate, 2),
        'overall_sla_compliance': round(overall_rate, 2)
    }

    return overall_sla_compliance