ㅇlibraries

    conda included libraries
    openbb
    pandas_ta : dataframe 이 ohlc 꼴로만 되있으면 알아서 해줌
    TA-Lib -> pandas_ta 가 일정부분 wrapping 하는 목적으로 같이 설치 / 직접 쓰기 불편함
    FinanceDataReader


ㅇ 모듈 외 할거

1. 주요 평균회귀 지표들 하나씩 콜매수 / 풋매수로 돌려보기 > 뭐가 제일 성과 좋은지
2. 해당 지표들 하나씩 가지고 다른 방향성 전략도 돌려보기
3. 단독으로 유의한 애들만 추려서 좀 더 overlap 해보기


ㅇentry_strat
    해당 모듈의 목표 : 진입 / 청산의 시점 (return index of each signals) 의 array 도출하는 것

psar

The SAR dots beneath the current market price point to an uptrend;
The SAR dots above the market price point to a downtrend;
Enter a position when the price penetrates the SAR – buy if the price crosses above the SAR and sell if the price crosses below the SAR;
You stop and reverse your position when the price crosses the SAR again.
You can aim to improve your Parabolic SAR strategy by using other indicators to aid your decision-making. For example, it can be useful to use a different trend indicator, such as the ADX, to establish that you are actually in a trending market, as opposed to a sideways moving market. It's important to note that the Parabolic SAR is not designed to work in a sideways market.

ㅁ완료
    변동성 short 전략 진입식별

ㅁ당장 해야할것
    1. 변동성 long 전략 working 하는 순간 식별
        전략은 뭘로? : 
            방향성 제거된 롱베가 : 캘린더 매도 / 행사가는 어떻게? > test

    2. 하락 후 적정 반등 시점 식별
        전략은 뭘로? : 
            숏베가/방향성
            베가중립/방향성


ㅁ추후 과제



ㅇstrat_backtest.py

ㅁ완료
    # dist_from_atm 을 atm 대비 벌어진 값으로 설정되게 -> 완료 V
    # dist_from_atm 을 atm 대비 벌어진 수준(%) 구해서 알아서 설정되게 -> 완료 V
    # dist_from_atm 을 해당 시점 델타 기준으로 알아서 설정되게 (델타 20/15 -> 행사가 알아서 선정)
    #---> 위에 세개 전부 혼용 가능하도록 (
    # (진입시 "0.04%에 긋고 / +7.5 행사 위에다가 매도 후 / 델타 0.05짜리로 외가헤지" 와 같은 전략 구현)
    # 내가옵션 일정 수준 이상 내가격 가면 거래안되서 0원으로 가격 비는거 처리
    # V 일일히 bsm 으로 계산하기 -> 없는 IV interpolate + extrapolate 은 그냥 fillna 로 제일 마지막 iv랑 동일
    # 진입 /청산시점의 변수화 : 현재 월요일 투자 고정 -> 특정일 list 별로 진입하는걸로 수정 (각종 TA 등등 별도로 날짜 추출)
    # 특정 전략 내에서도 leg 별로 익절 손절 따로 구현될수 있게 (래더스프레드 외가익절 / 크레딧스프레드 따로 두기 등)
    => 이건 그냥 전략을 두개로 쪼개서 접근
    (111의 경우 크레딧 스프레드로 백테스트 한번 + 네이키드 풋으로 백테스트 한번 해서 손익 합산)
    # 캘린더라이즈
    # 매매 통계 내기
    1) 총 매매 횟수
    2) 매매 승률
    3) 평균 이익
    4) 평균 손실
    5) 손익비 (금액/금액)
    6) 샤프

ㅁ 당장 해야할것

    # number_of_contracts 도 sizing 구현 : vix지수 수준에 따른 sizing : 한 trade size : 0 ~ 3까지?...
        # 현실적으로 매 signal 마다 포지션 들어가는데 그것가지 sizing 하는건 증거금 관리에 어긋남...
        # 한 주에 해야 할 수량 정하고 signal 마다 나누어서 진입 -> 0 ~ 5(max)
    # 병렬처리 코드로 바꾸기 : concurrent.futures / multithreading / asyncio


ㅁ 추후 과제

# 가격은 있는데, 애초에 종가가 븅신같이 거래된 케이스 -> 2021-02-24일 4월물 392.5 풋옵션. 혼자 5.62에 거래...
# 복리로 투자했으면 어떻게 됬을지? 누적수익률 구하는 함수


ㅇpreprocessing.py

ㅁ 당장 해야할것

ㅁ완료
# interp 가격 -> price nan/0에 밀어넣기 + dte = 1일때는 내재가치로 바꾸기 / 0.01 이하 가격 0.01로 통일시키기
# db에 테이블 밀어넣기 + 복수 테이블로 데이터들 파편화하는 구조 생각해보기

ㅁ해야할것
데이터 포맷 바뀌게되는 상황 + 신규 데이터 들어오는거 로데이터 어떻게 업데이트할지 재차 고민 (preprocessing 모듈의 1/2번 단락들)

ㅇ option_calc.py

ㅁ 스큐 / 텀 구하는 식 수정하기