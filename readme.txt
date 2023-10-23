Things I need

conda included libraries
openbb

pandas_ta : dataframe 이 ohlc 꼴로만 되있으면 df.ta.기술지표(파라메터) 형식으로 굳이 데이터 지정 안 해줘도 알아서 해줌
TA-Lib -> pandas_ta 가 일정부분 wrapping 하는 목적으로 같이 설치 / 직접 쓰기 불편함

FinanceDataReader



ㅇstrat_backtest.py

ㅁ완료
# dist_from_atm 을 atm 대비 벌어진 값으로 설정되게 -> 완료 V
# dist_from_atm 을 atm 대비 벌어진 수준(%) 구해서 알아서 설정되게 -> 완료 V
# dist_from_atm 을 해당 시점 델타 기준으로 알아서 설정되게 (델타 20/15 -> 행사가 알아서 선정)
#---> 위에 세개 전부 혼용 가능하도록 (
# (진입시 "0.04%에 긋고 / +7.5 행사 위에다가 매도 후 / 델타 0.05짜리로 외가헤지" 와 같은 전략 구현)
# 내가옵션 일정 수준 이상 내가격 가면 거래안되서 0원으로 가격 비는거 처리
# V 일일히 bsm 으로 계산하기 -> 없는 IV interpolate + extrapolate 은 그냥 fillna 로 제일 마지막 iv랑 동일

ㅁ해야할것
# 현재 월요일 투자 고정 -> 특정일 list 별로 진입하는걸로 수정 (각종 TA 등등 별도로 날짜 추출)

# 콜풋 따로 계산할게 아니라 수익률 합해서 profit taking / stop loss 한번에 구현
# leg 별로 익절 손절 따로 구현될수 있게 (래더스프레드 외가익절 / 크레딧스프레드 따로 두기 등)
# 캘린더 매매
# number_of_contracts vol-based dynamic sizing 구현
# 복리로 투자했으면 어떻게 됬을지? 누적수익률 구하는 함수


ㅇpreprocessing.py

ㅁ완료
# interp 가격 -> price nan/0에 밀어넣기 + dte = 1일때는 내재가치로 바꾸기 / 0.01 이하 가격 0.01로 통일시키기
# db에 테이블 밀어넣기 + 복수 테이블로 데이터들 파편화하는 구조 생각해보기

ㅁ해야할것


