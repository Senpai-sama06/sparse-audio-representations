import torch
from lrac.metrics import waveform_metrics, summarize
from lrac.losses import masked_mse


def test_metrics_detect_amplitude_error_and_handle_silence():
    x = torch.sin(torch.arange(2048).float() * .1)[None, None]
    perfect = waveform_metrics(x, x, 16000)
    scaled = waveform_metrics(x, x * .5, 16000)
    assert perfect["snr_db"] > 100
    assert 5.9 < scaled["snr_db"] < 6.1
    assert scaled["si_sdr_db"] > scaled["snr_db"] + 20
    assert waveform_metrics(x * 0, x * 0, 16000)["silence"]


def test_padding_is_not_counted_in_mse():
    x, y = torch.zeros(2, 1, 10), torch.zeros(2, 1, 10)
    y[0, 0, 5:] = 100
    assert masked_mse(x, y, torch.tensor([5, 10])) == 0
def test_speech_metrics_report_invalid_short_reference():
    from lrac.metrics import speech_quality_metrics
    import torch
    result = speech_quality_metrics(torch.zeros(64), torch.zeros(64), 16000)
    assert result["pesq_wb"] is None and result["stoi"] is None
    assert "too short or silent" in result["pesq_wb_status"]
    assert result["visqol"] is None
