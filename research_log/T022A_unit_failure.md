# T022-A implementation increment failure and minimal repair

2026-09-13 ~23:39+08. Unit release20260913-233843-taisp-t022a-spatial-unit.
Command: absolute project .venv/bin/python -m pytest -q tests/test_spatial_dose.py
tests/test_common_jacobian.py tests/test_spatial_action.py tests/test_trust_radius.py.
Result:3failed,16passed,1skipped in2.79s. No real-model parity or AP run occurred.

Failure: RuntimeError: DispatchKey FuncTorchGradWrapper doesn't correspond to a device.
At spatial.regional_gradients -> torch.func.jvp(lambda p:isp(image.detach(),p)) ->
ISP ops.gamma -> image.new_tensor(1e-6). The newly ported helper detached the image
inside the transformed lambda; the validated A1 donor detaches outside the lambda.
Minimal repair restores the donor convention: x=image.detach() before JVP; lambda
p:isp(x,p). No ISP, loss, formula, tolerance, dtype or model change. Rerun focused
tests before integrating any driver. This is an implementation error, not AP evidence.
