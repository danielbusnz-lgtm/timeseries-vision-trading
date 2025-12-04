import matplotlib.pyplot as plt
from data_fetcher import StockDataFetcher, GramianAngularField

fetcher = StockDataFetcher('AAPL')
df = fetcher.fetch()
window = df['close'].iloc[0:60].values

gaf = GramianAngularField(method='summation', image_size=64)
image = gaf.transform(window)


plt.imshow(image, cmap='viridis')
plt.colorbar()
plt.title('GAF Image  of 60-day price window')
plt.show()
plt.savefig('gaf_visualization.png')
print("Image saved to gaf_visualization.png")
