import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from frame_timeline import FrameTimeline


class TestWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Frame Timeline Test")
        self.resize(1200, 200)

        self.timeline = FrameTimeline()
        self.setCentralWidget(self.timeline)

        path = "/Users/dvuglaf/aaaaaaaaaaaaaaaaaaaaaaaaaaa.mp4"
        self.timeline.load_video(path)
        # self.open_video()

    def open_video(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video (*.mp4 *.avi *.mkv)"
        )

        if path:
            self.timeline.load_video(path)

            # Demo: random filter
            import numpy as np

            mask = np.random.rand(
                self.timeline.total_frames
            ) > 0.3

            self.timeline.set_mask(mask)


def main():

    app = QApplication(sys.argv)

    win = TestWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()