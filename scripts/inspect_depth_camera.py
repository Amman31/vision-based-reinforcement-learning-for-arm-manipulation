from pxr import Usd

usd_path = (
    "/home/amman31/isaac_lab/thesis_rl/"
    "src/thesis_rl/assets/depth_camera.usd"
)

stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)

print("\n=== Prim hierarchy inside depth_camera.usd ===\n")

for prim in stage.Traverse():
    print(
        f"Path: {prim.GetPath()} | "
        f"Type: {prim.GetTypeName()}"
    )

print("\n=== Camera prims only ===\n")

for prim in stage.Traverse():
    if prim.GetTypeName() == "Camera":
        print(f"CAMERA: {prim.GetPath()}")