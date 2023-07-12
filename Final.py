import streamlit as st
import wbgapi as wb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Define country codes
country_codes = {
    'United States': 'USA',
    'China': 'CHN',
    'European Union': 'EUU',
    'Japan': 'JPN',
    'Germany': 'DEU',
    'India': 'IND',
    'United Kingdom': 'GBR',
    'France': 'FRA',
    'Italy': 'ITA',
    'Brazil': 'BRA',
    'South Korea': 'KOR',
    'Australia': 'AUS',
    'Spain': 'ESP',
    'Mexico': 'MEX',
    'Indonesia': 'IDN',
    'Netherlands': 'NLD',
    'Saudi Arabia': 'SAU',
    'Turkey': 'TUR',
    'Switzerland': 'CHE',
    'Sweden': 'SWE',
    'Belgium': 'BEL',
    'Argentina': 'ARG',
    'Thailand': 'THA',
    'Nigeria': 'NGA',
    'Iran': 'IRN',
    'Austria': 'AUT',
    'Egypt': 'EGY',
    'High Income': 'HIC',
    'Middle Income': 'MIC',
    'World': 'WLD'
}

series_list = ['NY.GDP.PCAP.KD', 'SP.POP.TOTL']

# Streamlit input to select countries
selected_countries = st.multiselect('Select countries', list(country_codes.keys()))

# Convert to country codes
selected_country_codes = [country_codes[country] for country in selected_countries]

end_model = st.slider('Select end year for the model', 1980, 2100, 2030) + 1

def preprocess_data(data):
    data_transposed = data.transpose()
    data_transposed.reset_index(inplace=True)
    data_transposed.columns = ['year', 'value']
    data_transposed['year'] = data_transposed['year'].apply(lambda x: int(x[2:]))
    return data_transposed['year'].values.reshape(-1,1), data_transposed['value'].values.reshape(-1,1)

models = [
    ("Ridge Regression", Ridge(random_state=0)),
    ("Polynomial Regression", make_pipeline(PolynomialFeatures(), LinearRegression())),
]

def select_best_model(X, y, models):
    best_score, best_model = -np.inf, None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    for name, model in models:
        model.fit(X_train, y_train.ravel())
        score = model.score(X_val, y_val)
        if score > best_score:
            best_score, best_model = score, (name, model)
    return best_model, X_val, y_val

def plot_data(X, y, future_years, future_pred, title, country, ax):
    ax.scatter(X, y, label=f'Real Value ({country})')
    ax.plot(future_years, future_pred, label=f'Predicted Value ({country})')

predictions = {}

# Create matplotlib Figure and Axes
fig1, ax1 = plt.subplots(figsize=(12,6))
fig2, ax2 = plt.subplots(figsize=(12,6))

for series in series_list:
    if series == 'NY.GDP.PCAP.KD':
        ax = ax1
    else:
        ax = ax2
    
    for country_code in selected_country_codes:
        data = wb.data.DataFrame(series, country_code, time=range(1980, end_model))
        X, y = preprocess_data(data)
        (best_model_name, best_model), X_val, y_val = select_best_model(X, y, models)
        y_pred = best_model.predict(X_val)
        future_years = np.arange(1980, end_model).reshape(-1,1)
        future_pred = best_model.predict(future_years)
        if country_code not in predictions:
            predictions[country_code] = {}
        predictions[country_code][series] = future_pred
        plot_data(X, y, future_years, future_pred, f'{series} in {country_code} (1980-{int(end_model)-1})', country_code, ax)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.legend()  # Add legend to the plot

# Set titles
ax1.set_title('GDP per capita in different countries (1980-{})'.format(int(end_model)-1))
ax2.set_title('Population in different countries (1980-{})'.format(int(end_model)-1))

st.pyplot(fig1)
st.pyplot(fig2)

# GDP plot
fig3, ax3 = plt.subplots(figsize=(12,6))
for country_code, data in predictions.items():
    country_name = [key for key, value in country_codes.items() if value == country_code][0]
    result = data['NY.GDP.PCAP.KD'].ravel() * data['SP.POP.TOTL'].ravel() / 1e12  # Adjust for trillions
    ax3.plot(range(1980, end_model), result, label=country_name)

ax3.set_title('GDP of the selected Countries (1980-{})'.format(int(end_model)-1))
ax3.set_xlabel('Year')
ax3.set_ylabel('GDP (in Trillions)')
ax3.legend()  # Add legend to the plot

st.pyplot(fig3)

# Display model details after the plots
for series in series_list:
    for country_code in selected_country_codes:
        data = wb.data.DataFrame(series, country_code, time=range(1980, end_model))
        X, y = preprocess_data(data)
        (best_model_name, best_model), X_val, y_val = select_best_model(X, y, models)
        y_pred = best_model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        st.write(f"The best model for series {series} in country {country_code} is: {best_model_name}")
        st.write(f"R^2 Score: {r2}")