from model_manager import ModelManager


def main():
    metrics = ModelManager().train_model()
    print(metrics)


if __name__ == "__main__":
    main()
