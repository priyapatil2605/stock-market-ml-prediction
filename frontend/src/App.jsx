import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [comparison, setComparison] = useState([]);
  const [backtest, setBacktest] = useState([]);
  const [explain, setExplain] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);

      const [comparisonRes, backtestRes, explainRes] =
        await Promise.all([
          axios.get(`${API}/compare`),
          axios.get(`${API}/backtest`),
          axios.get(`${API}/explain`),
        ]);

      setComparison(comparisonRes.data);
      setBacktest(backtestRes.data);
      setExplain(explainRes.data.feature_importance);
    } catch (err) {
      console.error(err);
      setError(
        "Could not connect to the backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="center">
        <h2>Loading Stock Market ML Dashboard...</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div className="center">
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={loadData}>Try Again</button>
      </div>
    );
  }

  return (
    <div className="app">

      <header>
        <div>
          <h1>Stock Market ML Dashboard</h1>
          <p>
            Machine Learning Based Stock Movement Prediction and Backtesting
          </p>
        </div>

        <button onClick={loadData}>
          Refresh Data
        </button>
      </header>


      <section>
        <h2>Model Comparison</h2>

        <div className="cards">
          {comparison.map((model) => (
            <div className="card" key={model.Model}>
              <h3>{model.Model}</h3>

              <div className="metric">
                <span>Accuracy</span>
                <strong>{model.Accuracy.toFixed(2)}%</strong>
              </div>

              <div className="metric">
                <span>F1 Score</span>
                <strong>{model["F1 Score"].toFixed(2)}%</strong>
              </div>

              <div className="metric">
                <span>Precision</span>
                <strong>{model.Precision.toFixed(2)}%</strong>
              </div>
            </div>
          ))}
        </div>
      </section>


      <section>
        <h2>Backtesting Performance</h2>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Annualized Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
              </tr>
            </thead>

            <tbody>
              {backtest.map((item) => (
                <tr key={item.Strategy}>
                  <td>{item.Strategy}</td>
                  <td>{item["Total Return (%)"]}%</td>
                  <td>{item["Annualized Return (%)"]}%</td>
                  <td>{item["Sharpe Ratio"]}</td>
                  <td>{item["Max Drawdown (%)"]}%</td>
                  <td>{item["Win Rate (%)"]}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>


      <section>
        <h2>AI Model Explainability</h2>
        <p className="subtitle">
          SHAP Feature Importance for XGBoost
        </p>

        <div className="feature-list">
          {explain.slice(0, 10).map((item) => {
            const maxValue = explain[0]?.Mean_Absolute_SHAP || 1;

            const width =
              (item.Mean_Absolute_SHAP / maxValue) * 100;

            return (
              <div className="feature" key={item.Feature}>
                <div className="feature-header">
                  <span>{item.Feature}</span>

                  <strong>
                    {item.Mean_Absolute_SHAP.toFixed(4)}
                  </strong>
                </div>

                <div className="bar-background">
                  <div
                    className="bar"
                    style={{ width: `${width}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </section>


      <section className="project-summary">
        <h2>Project Summary</h2>

        <div className="summary-grid">
          <div>
            <strong>Models</strong>
            <p>Logistic Regression, XGBoost, LSTM</p>
          </div>

          <div>
            <strong>Validation</strong>
            <p>Time-Based Split + Walk-Forward Validation</p>
          </div>

          <div>
            <strong>Explainability</strong>
            <p>SHAP Feature Importance</p>
          </div>

          <div>
            <strong>Backtesting</strong>
            <p>Portfolio-Level Strategy Evaluation</p>
          </div>
        </div>
      </section>

    </div>
  );
}

export default App;