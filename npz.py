Python 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license()" for more information.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 7))

img = ax.imshow(frames[0], cmap="inferno", animated=True)
cbar = plt.colorbar(img, ax=ax, label="Raw Temperature Value")

title = ax.set_title("Frame: 0 / {}".format(len(frames)))

def update(frame_index):
    img.set_data(frames[frame_index])
    title.set_text(f"Frame: {frame_index} / {len(frames)}")
    return img, title

ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=33,
    blit=False,
    repeat=True
)

plt.show()
