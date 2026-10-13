# OpenVLA evaluation utilities

`event_sae/openvla/eval/` adapts utilities from the official
[OpenVLA repository](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot):

- `openvla_utils.py`: image cropping, model inference and the system prompt.
- `robot_utils.py`: seeding, device selection and gripper action conversion.
- `libero/libero_utils.py`: LIBERO environment and image utilities.

OpenVLA is distributed under the [MIT license](LICENSE), with the upstream
copyright notice retained. The code has been reorganized and adapted for Event-SAE.

`quat2axisangle` originates from
[robosuite](https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py),
as acknowledged by OpenVLA. Its upstream [license and notices](LICENSE.robosuite)
are retained here.
