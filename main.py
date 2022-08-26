from utils import load_data, clean_data, split_normalize_data, oversample_data, build_model, train_model, get_model_metrics

def main():
  df = load_data()
  df = clean_data(df)
  data = split_normalize_data(df)
  data = oversample_data(data)

  model = build_model()
  model = train_model(model, data)
  # Best Model is saved

  print('Test Metrics-')
  test_metrics = get_model_metrics(model, data)
  print(test_metrics)

if __name__ == '__main__':
    main()