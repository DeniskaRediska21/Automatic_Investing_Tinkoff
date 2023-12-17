from pandas import DataFrame
from tinkoff.invest import Client, InstrumentStatus, SharesResponse, InstrumentIdType
from tinkoff.invest.services import InstrumentsService, MarketDataService


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



token ='t.lL1iEp2MJ3VuLvGaK0wuWXVemPMppCA9OVoCBipcn1KYdcMIa_qncdGtP9PQrwGnWOVH-gzYcJmL51W82t_0IQ'
TICKER = "VKCO"
id='2059195636'

def run():
    with Client(token) as cl:
        instruments: InstrumentsService = cl.instruments
        market_data: MarketDataService = cl.market_data

        # r = instruments.share_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id="BBG004S683W7")
        # print(r)

        l = []
        for method in ['shares', 'bonds', 'etfs']: # , 'currencies', 'futures']:
            for item in getattr(instruments, method)().instruments:
                l.append({
                    'ticker': item.ticker,
                    'figi': item.figi,
                    'type': method,
                    'name': item.name,
                })

        df = DataFrame(l)
        # df.to_json()

        df = df[df['ticker'] == TICKER]
        if df.empty:
            print(f"Нет тикера {TICKER}")
            return

        # print(df.iloc[0])
        print(df['figi'].iloc[0])
        figi=df['figi'].iloc[0]
        with Client(token) as client:
            r = client.market_data.get_trading_status(figi=figi)
            print(r)
            print(r.market_order_available_flag)


if __name__ == '__main__':
    print(f"{TICKER}\n")
    run()
