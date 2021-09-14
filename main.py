from src.recorderapp import RecorderApp


def main():
    recorder_app = RecorderApp(t_freq=.03, device_id=0, block_len=.01)


if __name__ == "__main__":
    main()
