from quant_forecast_lab.experiment import sha256_file


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "data.txt"
    path.write_bytes(b"quant-forecast-lab\n")

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64
