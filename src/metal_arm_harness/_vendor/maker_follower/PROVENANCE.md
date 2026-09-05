# LeRobot Maker follower

Source: https://github.com/makermods-robotics/lerobot/tree/af9072a8ca4b1767ef25b98eed1149d1bbf89f46/src/lerobot/robots/maker_follower

Branch: `robot/makermods-maker-arm`, commit `af9072a8ca4b1767ef25b98eed1149d1bbf89f46`.
Apache-2.0; source copyright headers retained. See LICENSE in this folder.
Only sibling imports into LeRobot were made absolute. This compatibility copy
lets the harness use Maker and Metal together while their upstream robot classes
live on separate branches. The installed LeRobot RobstrideMotorsBus is reused.

The harness does not call the follower's connect/calibrate/read paths on real
hardware: their implicit enable, fault-clear reads, and stale feedback fallback
do not satisfy the harness contract. See arms/maker.py and docs/MAKER.md.
