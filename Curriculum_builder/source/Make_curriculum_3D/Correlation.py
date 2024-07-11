import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# CSV 파일 경로를 적절히 수정하세요.
csv_file_path = "points7_3.csv"
df = pd.read_csv(csv_file_path)
# feature들 간의 상관관계를 계산합니다.
corr = df.corr()

# seaborn의 heatmap을 사용하여 상관관계를 시각화합니다.
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()
