# Current project state

2026-09-13T07:01:40.4102208+08:00 — T013-G IN_PROGRESS under R021/3a1f363; T013-F CLOSED.
Plan21eaa31 pinned arrays and float32-derived reconstruction bound before outcomes.
Baseline5passed13.51s. New offline NumPyfloat64 factorization +synthetic tests ready.
Local focused test aborted at NumPy matmul; minimal NumPy+torch repro confirms
OMP Error15 duplicate libiomp5md.dll (numpy1.26.4/torch2.13CPU). Error retained;
no KMP_DUPLICATE_LIB_OK workaround. Use existing remoteCPU environment for tests.
No realmodel/newfeature/gradient execution or optimizer allowed/needed. All arrayssaved.
Next deploy, remoteCPU focused tests thenfullregression thenofflineaudit onlyif tests pass.
No formula/tolerance changes. StopafterT013G; no T013H/training/redesign/newdata/AP.
