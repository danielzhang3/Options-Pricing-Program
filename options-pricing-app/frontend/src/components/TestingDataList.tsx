import React, { useState, useEffect, useCallback } from 'react';
import './TestingDataList.css';

interface TestingOptionData {
  id: number;
  symbol: string;
  trade_datetime: string;
  quantity: number;
  trade_price: number;
  close_price: number | null;
  proceeds: number | null;
  comm_fee: number | null;
  basis: number | null;
  realized_pl: number | null;
  mtm_pl: number | null;
  code: string | null;
  underlying: string;
  strike_price: number;
  expiration_date: string;
  option_type: string;
  moneyness?: number;
}

const TestingDataList: React.FC = () => {
  const [data, setData] = useState<TestingOptionData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 20;

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      let url = `http://localhost:8000/api/testing-options/?page=${currentPage}&page_size=${itemsPerPage}`;
      if (filter) {
        url += `&underlying=${encodeURIComponent(filter)}`;
      }
      
      const response = await fetch(url);
      if (response.ok) {
        const result = await response.json();
        setData(result.results || result);
        // Calculate total pages if pagination info is available
        if (result.count) {
          setTotalPages(Math.ceil(result.count / itemsPerPage));
        }
      } else {
        throw new Error('Failed to fetch data');
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [currentPage, filter]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const formatCurrency = (value: number | null) => {
    if (value === null) return '-';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatDateTime = (dateTimeString: string) => {
    return new Date(dateTimeString).toLocaleString();
  };

  const handleFilterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFilter(e.target.value);
    setCurrentPage(1); // Reset to first page when filtering
  };

  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage);
  };

  if (loading) {
    return (
      <div className="testing-data-list">
        <h2>Testing Options Data</h2>
        <div className="loading">Loading data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="testing-data-list">
        <h2>Testing Options Data</h2>
        <div className="error-message">{error}</div>
        <button onClick={fetchData} className="retry-btn">Retry</button>
      </div>
    );
  }

  return (
    <div className="testing-data-list">
      <h2>Testing Options Data</h2>
      <p>View and analyze uploaded IBKR options trading data</p>

      {/* Filter Section */}
      <div className="filter-section">
        <input
          type="text"
          placeholder="Filter by underlying symbol (e.g., AAPL, TSLA)"
          value={filter}
          onChange={handleFilterChange}
          className="filter-input"
        />
        <button onClick={() => setFilter('')} className="clear-filter-btn">
          Clear Filter
        </button>
      </div>

      {/* Data Table */}
      {data.length === 0 ? (
        <div className="empty-state">
          <p>No testing data found. Upload a CSV file to see data here.</p>
        </div>
      ) : (
        <>
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Underlying</th>
                  <th>Type</th>
                  <th>Strike</th>
                  <th>Moneyness</th>
                  <th>Expiration</th>
                  <th>Trade Date</th>
                  <th>Quantity</th>
                  <th>Trade Price</th>
                  <th>Close Price</th>
                  <th>Proceeds</th>
                  <th>Realized P&L</th>
                  <th>MTM P&L</th>
                </tr>
              </thead>
              <tbody>
                {data.map((item) => (
                  <tr key={item.id}>
                    <td className="symbol-cell">{item.symbol}</td>
                    <td>{item.underlying}</td>
                    <td>
                      <span className={`option-type ${item.option_type}`}>
                        {item.option_type.toUpperCase()}
                      </span>
                    </td>
                    <td>{formatCurrency(item.strike_price)}</td>
                    <td>
                      {item.moneyness ? (
                        <span className={item.moneyness > 1 ? 'in-the-money' : item.moneyness < 1 ? 'out-of-the-money' : 'at-the-money'}>
                          {item.moneyness.toFixed(3)}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td>{formatDate(item.expiration_date)}</td>
                    <td>{formatDateTime(item.trade_datetime)}</td>
                    <td className={item.quantity > 0 ? 'positive' : 'negative'}>
                      {item.quantity}
                    </td>
                    <td>{formatCurrency(item.trade_price)}</td>
                    <td>{formatCurrency(item.close_price)}</td>
                    <td>{formatCurrency(item.proceeds)}</td>
                    <td className={item.realized_pl && item.realized_pl > 0 ? 'positive' : 'negative'}>
                      {formatCurrency(item.realized_pl)}
                    </td>
                    <td className={item.mtm_pl && item.mtm_pl > 0 ? 'positive' : 'negative'}>
                      {formatCurrency(item.mtm_pl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="pagination">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className="page-btn"
              >
                Previous
              </button>
              
              <span className="page-info">
                Page {currentPage} of {totalPages}
              </span>
              
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="page-btn"
              >
                Next
              </button>
            </div>
          )}

          {/* Summary Stats */}
          <div className="summary-stats">
            <div className="stat-card">
              <h4>Total Records</h4>
              <div className="stat-value">{data.length}</div>
            </div>
            <div className="stat-card">
              <h4>Total Realized P&L</h4>
              <div className="stat-value">
                {formatCurrency(
                  data.reduce((sum, item) => sum + (item.realized_pl || 0), 0)
                )}
              </div>
            </div>
            <div className="stat-card">
              <h4>Total MTM P&L</h4>
              <div className="stat-value">
                {formatCurrency(
                  data.reduce((sum, item) => sum + (item.mtm_pl || 0), 0)
                )}
              </div>
            </div>
            <div className="stat-card">
              <h4>Avg Moneyness</h4>
              <div className="stat-value">
                {(() => {
                  const validMoneyness = data.filter(item => item.moneyness !== null && item.moneyness !== undefined);
                  if (validMoneyness.length === 0) return '-';
                  const avg = validMoneyness.reduce((sum, item) => sum + (item.moneyness || 0), 0) / validMoneyness.length;
                  return avg.toFixed(3);
                })()}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default TestingDataList; 