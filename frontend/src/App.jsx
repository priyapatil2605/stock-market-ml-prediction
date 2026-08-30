import { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

const STRATEGY_COLORS = {
  "Logistic Regression": "#e8a33d",
  "XGBoost": "#5b9bd5",
  "Equal Weight Benchmark": "#7a8290",
};

function App() {
  const [comparison, setComparison] = useState([]);
  const [backtest, setBacktest] = useState([]);
  const [explain, setExplain] = useState([]);
  const [equityCurve, setEquityCurve] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      setError("");

      const [comparisonRes, backtestRes, explainRes, equityRes] =
        await Promise.all([
          axios.get(`${API}/compare`),
          axios.get(`${API}/backtest`),
          axios.get(`${API}/explain`),
          axios.get(`${API}/equity-curve`),
        ]);

      setComparison(comparisonRes.data);
      setBacktest(backtestRes.data);
      setExplain(explainRes.data.feature_importance);
      setEquityCurve(equityRes.data);
      setLastUpdated(new Date());
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
      <div className="statusScreen">
        <div className="pulseDot" />
        <p className="mono">LOADING RESEARCH DATA…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="statusScreen">
        <p className="mono errorLabel">CONNECTION ERROR</p>
        <p className="errorMessage">{error}</p>
        <button className="btnPrimary" onClick={loadData}>
          Retry Connection
        </button>
      </div>
    );
  }

  const bestModel = [...comparison].sort(
    (a, b) => b["F1 Score"] - a["F1 Score"]
  )[0];

  return (
    <div className="app">
      {/* Status strip — the whole point: this tells you plainly it's static research, not a live feed */}
      <div className="statusStrip">
        <div className="statusStripInner">
          <div className="statusItem">
            <span className="statusDotStatic" />
            <span className="mono statusLabel">
              DATA MODE — HISTORICAL / STATIC BACKTEST
              {equityCurve &&
                ` · COVERS ${equityCurve.data_start} → ${equityCurve.data_end}`}
            </span>
          </div>
          <div className="statusItem statusItemRight">
            <span className="mono statusLabel">
              {lastUpdated
                ? `LAST LOADED ${lastUpdated.toLocaleTimeString()}`
                : ""}
            </span>
          </div>
        </div>
      </div>

      <header className="header">
        <div>
          <p className="mono eyebrow">QUANT RESEARCH TERMINAL</p>
          <h1 className="title">Alpha Signal</h1>
          <p className="subtitle">
            ML-based equity movement prediction, walk-forward validated,
            backtested net of transaction costs
          </p>
        </div>
        <button className="btnPrimary" onClick={loadData}>
          Refresh Data
        </button>
      </header>

      {/* MODEL COMPARISON */}
      <section className="section">
        <div className="sectionHeading">
          <h2>Model Comparison</h2>
          <p className="sectionSub">Held-out test performance, by model</p>
        </div>

        <div className="cardGrid">
          {comparison.map((model) => (
            <div
              className={
                "modelCard" +
                (model.Model === bestModel?.Model ? " modelCardBest" : "")
              }
              key={model.Model}
            >
              <div className="modelCardHead">
                <h3>{model.Model}</h3>
                {model.Model === bestModel?.Model && (
                  <span className="badge">BEST F1</span>
                )}
              </div>

              <div className="metricRow">
                <span className="metricLabel">Accuracy</span>
                <span className="metricValue mono">
                  {model.Accuracy.toFixed(2)}%
                </span>
              </div>
              <div className="metricRow">
                <span className="metricLabel">F1 Score</span>
                <span className="metricValue mono">
                  {model["F1 Score"].toFixed(2)}%
                </span>
              </div>
              <div className="metricRow">
                <span className="metricLabel">Precision</span>
                <span className="metricValue mono">
                  {model.Precision.toFixed(2)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* EQUITY CURVE */}
      <section className="section">
        <div className="sectionHeading">
          <h2>Equity Curve</h2>
          <p className="sectionSub">
            Portfolio value over time, starting at 1.0 — net of transaction
            costs, {equityCurve?.data_start} to {equityCurve?.data_end}
          </p>
        </div>

        <div className="chartCard">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={equityCurve?.points || []}>
              <CartesianGrid stroke="#1d2128" vertical={false} />
              <XAxis
                dataKey="Date"
                tick={{ fill: "#7a8290", fontSize: 11, fontFamily: "IBM Plex Mono" }}
                tickLine={false}
                axisLine={{ stroke: "#262b33" }}
                minTickGap={60}
              />
              <YAxis
                tick={{ fill: "#7a8290", fontSize: 11, fontFamily: "IBM Plex Mono" }}
                tickLine={false}
                axisLine={{ stroke: "#262b33" }}
                tickFormatter={(v) => v.toFixed(2)}
                width={48}
              />
              <Tooltip
                contentStyle={{
                  background: "#14171c",
                  border: "1px solid #262b33",
                  borderRadius: 6,
                  fontFamily: "IBM Plex Mono",
                  fontSize: 12,
                }}
                labelStyle={{ color: "#7a8290" }}
                formatter={(value) => value.toFixed(4)}
              />
              <Legend
                wrapperStyle={{
                  fontSize: 12,
                  fontFamily: "IBM Plex Sans",
                  paddingTop: 12,
                }}
              />
              {Object.keys(STRATEGY_COLORS).map((key) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={STRATEGY_COLORS[key]}
                  strokeWidth={key === "Equal Weight Benchmark" ? 1.5 : 2}
                  strokeDasharray={
                    key === "Equal Weight Benchmark" ? "4 3" : undefined
                  }
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* BACKTEST */}
      <section className="section">
        <div className="sectionHeading">
          <h2>Backtesting Performance</h2>
          <p className="sectionSub">
            Net of assumed transaction costs — compared against an
            equal-weight benchmark, not a strawman
          </p>
        </div>

        <div className="tableWrap">
          <table className="dataTable">
            <thead>
              <tr>
                <th>Strategy</th>
                <th className="num">Total Return</th>
                <th className="num">Annualized</th>
                <th className="num">Sharpe</th>
                <th className="num">Max Drawdown</th>
                <th className="num">Win Rate</th>
              </tr>
            </thead>
            <tbody>
              {backtest.map((item) => {
                const isBenchmark = item.Strategy
                  .toLowerCase()
                  .includes("benchmark");
                const totalReturn = item["Total Return (%)"];
                const sharpe = item["Sharpe Ratio"];
                return (
                  <tr
                    key={item.Strategy}
                    className={isBenchmark ? "benchmarkRow" : ""}
                  >
                    <td className="strategyCell">
                      {item.Strategy}
                      {isBenchmark && (
                        <span className="benchmarkTag">BENCHMARK</span>
                      )}
                    </td>
                    <td className={`num mono ${totalReturn >= 0 ? "pos" : "neg"}`}>
                      {totalReturn >= 0 ? "+" : ""}
                      {totalReturn}%
                    </td>
                    <td className="num mono">
                      {item["Annualized Return (%)"]}%
                    </td>
                    <td className={`num mono ${sharpe >= 0 ? "pos" : "neg"}`}>
                      {sharpe}
                    </td>
                    <td className="num mono neg">
                      {item["Max Drawdown (%)"]}%
                    </td>
                    <td className="num mono">{item["Win Rate (%)"]}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* SHAP EXPLAINABILITY */}
      <section className="section">
        <div className="sectionHeading">
          <h2>Model Explainability</h2>
          <p className="sectionSub">
            SHAP mean absolute feature importance — XGBoost
          </p>
        </div>

        <div className="featureList">
          {explain.slice(0, 10).map((item, i) => {
            const maxValue = explain[0]?.Mean_Absolute_SHAP || 1;
            const width = (item.Mean_Absolute_SHAP / maxValue) * 100;
            return (
              <div className="featureRow" key={item.Feature}>
                <span className="mono featureRank">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="featureName">{item.Feature}</span>
                <div className="featureBarTrack">
                  <div
                    className="featureBarFill"
                    style={{ width: `${width}%` }}
                  />
                </div>
                <span className="mono featureValue">
                  {item.Mean_Absolute_SHAP.toFixed(4)}
                </span>
              </div>
            );
          })}
        </div>
      </section>

      {/* RESEARCH NOTE — answers "is this live / is this financial advice" directly */}
      <section className="section">
        <div className="researchNote">
          <p className="mono researchNoteLabel">RESEARCH NOTE</p>
          <p>
            This dashboard reflects a historical backtest, not a live trading
            signal. Predictions are generated from a fixed, walk-forward
            validated model trained on past data — nothing here updates in
            real time, and nothing here is investment advice. The value of
            this project is the rigor of the validation methodology, not a
            claim that it beats the market.
          </p>
        </div>

        <div className="summaryGrid">
          <div className="summaryItem">
            <span className="mono summaryLabel">MODELS</span>
            <p>Logistic Regression · XGBoost · LSTM</p>
          </div>
          <div className="summaryItem">
            <span className="mono summaryLabel">VALIDATION</span>
            <p>Walk-forward, time-based splits — no lookahead bias</p>
          </div>
          <div className="summaryItem">
            <span className="mono summaryLabel">EXPLAINABILITY</span>
            <p>SHAP feature importance</p>
          </div>
          <div className="summaryItem">
            <span className="mono summaryLabel">BACKTESTING</span>
            <p>Portfolio-level, net of transaction costs</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
