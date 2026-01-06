"""
SQLAlchemy ORM 모델 정의
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    SmallInteger,
    Float,
    DateTime,
    String,
    PrimaryKeyConstraint,
    Index,
)
from sqlalchemy.dialects.mssql import TINYINT

from src.data.db_connector import Base


class DS2PrimQtPrc(Base):
    """
    DS2 Primary Quote Price 테이블
    
    주식의 일별 시세 데이터를 저장하는 테이블
    - OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터
    - Bid/Ask 호가 데이터
    - VWAP (거래량가중평균가격) 등
    """
    __tablename__ = "DS2PrimQtPrc"
    __table_args__ = (
        PrimaryKeyConstraint("InfoCode", "MarketDate", name="pkey_DS2PrimQtPrc"),
        Index(
            "DS2PrimQtPrc_1",
            "InfoCode",
            "ExchIntCode",
            "MarketDate",
        ),
        {"schema": "dbo"},
    )

    # Primary Key 컬럼
    InfoCode: int = Column(Integer, nullable=False, comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)")
    MarketDate: datetime = Column(DateTime, nullable=False, comment="거래일")

    # 거래소 및 통화 정보
    ExchIntCode: Optional[int] = Column(SmallInteger, nullable=True, comment="거래소 코드")
    ISOCurrCode: Optional[str] = Column(String(3), nullable=True, comment="ISO 통화 코드")
    RefPrcTypCode: Optional[int] = Column(TINYINT, nullable=True, comment="가격 확정 여부 (1=확정, 2=잠정)")

    # OHLCV 데이터
    Open_: Optional[float] = Column("Open_", Float, nullable=True, comment="시가")
    High: Optional[float] = Column(Float, nullable=True, comment="고가")
    Low: Optional[float] = Column(Float, nullable=True, comment="저가")
    Close_: Optional[float] = Column("Close_", Float, nullable=True, comment="종가")
    Volume: Optional[float] = Column(Float, nullable=True, comment="미수정 거래량")

    # 호가 데이터 (장마감 시점)
    Bid: Optional[float] = Column(Float, nullable=True, comment="장마감 시 매수 호가")
    Ask: Optional[float] = Column(Float, nullable=True, comment="장마감 시 매도 호가")

    # 추가 가격 데이터
    VWAP: Optional[float] = Column(Float, nullable=True, comment="거래량가중평균가격")
    MostTrdPrc: Optional[float] = Column(Float, nullable=True, comment="최다 거래 가격")
    ConsolVol: Optional[float] = Column(Float, nullable=True, comment="통합 거래량 (일본/독일/인도 복수거래소)")
    MostTrdVol: Optional[float] = Column(Float, nullable=True, comment="최다 거래량")

    # 메타 정보
    LicFlag: Optional[int] = Column(SmallInteger, nullable=True, comment="라이선스 비트맵 플래그")

    def __repr__(self) -> str:
        return (
            f"<DS2PrimQtPrc(InfoCode={self.InfoCode}, "
            f"MarketDate={self.MarketDate}, "
            f"Close_={self.Close_})>"
        )


class DS2PrimQtRI(Base):
    """
    DS2 Primary Quote Return Index 테이블
    
    주식의 수익률 지수(Return Index) 데이터를 저장하는 테이블
    - 배당 재투자를 반영한 총수익률 지수
    """
    __tablename__ = "DS2PrimQtRI"
    __table_args__ = (
        PrimaryKeyConstraint("InfoCode", "MarketDate", name="pkey_DS2PrimQtRI"),
        Index(
            "DS2PrimQtRI_1",
            "InfoCode",
            "ExchIntCode",
            "MarketDate",
        ),
        {"schema": "dbo"},
    )

    # Primary Key 컬럼
    InfoCode: int = Column(Integer, nullable=False, comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)")
    MarketDate: datetime = Column(DateTime, nullable=False, comment="거래일")

    # 거래소 정보
    ExchIntCode: Optional[int] = Column(SmallInteger, nullable=True, comment="거래소 코드")

    # 수익률 지수
    RI: Optional[float] = Column(Float, nullable=True, comment="수익률 지수 (Return Index)")

    # 메타 정보
    LicFlag: Optional[int] = Column(SmallInteger, nullable=True, comment="라이선스 비트맵 플래그")

    def __repr__(self) -> str:
        return (
            f"<DS2PrimQtRI(InfoCode={self.InfoCode}, "
            f"MarketDate={self.MarketDate}, "
            f"RI={self.RI})>"
        )


class DS2CtryQtInfo(Base):
    """
    DS2 Country Quote Info 테이블
    
    종목의 기본 정보를 저장하는 마스터 테이블
    - 종목 코드, 명칭, 지역, 통화 등 메타데이터
    """
    __tablename__ = "DS2CtryQtInfo"
    __table_args__ = (
        PrimaryKeyConstraint("InfoCode", name="pkey_DS2CtryQtInfo"),
        Index("DS2CtryQtInfo_1", "DsSecCode"),
        Index("DS2CtryQtInfo_2", "DsCode"),
        Index("DS2CtryQtInfo_3", "Region"),
        Index("DS2CtryQtInfo_4", "DsLocalCode"),
        Index("DS2CtryQtInfo_5", "DsQtName"),
        {"schema": "dbo"},
    )

    # Primary Key
    InfoCode: int = Column(Integer, nullable=False, comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)")

    # Datastream 코드 정보
    DsCode: Optional[str] = Column(String(13), nullable=True, comment="Datastream 코드")
    DsSecCode: Optional[int] = Column(Integer, nullable=True, comment="Datastream 증권 코드")
    DsLocalCode: Optional[str] = Column(String(13), nullable=True, comment="Datastream 로컬 코드")
    DsMnem: Optional[str] = Column(String(13), nullable=True, comment="Datastream 니모닉")
    DsQtName: Optional[str] = Column(String(91), nullable=True, comment="Datastream 종목명")

    # 지역 및 분류 정보
    Region: Optional[str] = Column(String(7), nullable=True, comment="지역 코드")
    RegCodeTypeId: Optional[int] = Column(SmallInteger, nullable=True, comment="지역 코드 유형 ID")
    TypeCode: Optional[str] = Column(String(5), nullable=True, comment="종목 유형 코드")

    # 상태 정보
    IsPrimQt: Optional[int] = Column(TINYINT, nullable=True, comment="주요 시세 여부")
    CovergCode: Optional[str] = Column(String(1), nullable=True, comment="커버리지 코드")
    StatusCode: Optional[str] = Column(String(1), nullable=True, comment="상태 코드")

    # 식별자 및 통화
    PermID: Optional[str] = Column(String(11), nullable=True, comment="영구 식별자")
    PrimISOCurrCode: Optional[str] = Column(String(3), nullable=True, comment="기본 ISO 통화 코드")

    # 상장폐지 정보
    DelistDate: Optional[datetime] = Column(DateTime, nullable=True, comment="상장폐지일")

    def __repr__(self) -> str:
        return (
            f"<DS2CtryQtInfo(InfoCode={self.InfoCode}, "
            f"DsQtName={self.DsQtName}, "
            f"Region={self.Region})>"
        )


class DS2CapEvent(Base):
    """
    DS2 Capital Event 테이블
    
    자본 이벤트 정보를 저장하는 테이블
    - 액면분할, 합병, 유상증자 등 기업 이벤트
    """
    __tablename__ = "DS2CapEvent"
    __table_args__ = (
        PrimaryKeyConstraint("InfoCode", "EventNum", name="pkey_DS2CapEvent"),
        Index("DS2CapEvent_1", "InfoCode", "EffectiveDate"),
        Index("DS2CapEvent_2", "InfoCode", "ActionTypeCode", "EffectiveDate"),
        Index("DS2CapEvent_3", "InfoCode", "AnnouncedDate"),
        Index("DS2CapEvent_4", "InfoCode", "ActionTypeCode", "AnnouncedDate"),
        Index("DS2CapEvent_5", "InfoCode", "RecordDate"),
        Index("DS2CapEvent_6", "InfoCode", "ActionTypeCode", "RecordDate"),
        Index("DS2CapEvent_7", "InfoCode", "ExpiryDate"),
        Index("DS2CapEvent_8", "InfoCode", "ActionTypeCode", "ExpiryDate"),
        {"schema": "dbo"},
    )

    # Primary Key
    InfoCode: int = Column(Integer, nullable=False, comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)")
    EventNum: int = Column(Integer, nullable=False, comment="이벤트 번호")

    # 이벤트 유형 정보
    ActionTypeCode: Optional[str] = Column(String(4), nullable=True, comment="액션 유형 코드")
    EventStatusCode: Optional[str] = Column(String(4), nullable=True, comment="이벤트 상태 코드")

    # 관련 종목 정보
    ResInfoCode: Optional[int] = Column(Integer, nullable=True, comment="결과 InfoCode")
    RenMarker: Optional[str] = Column(String(1), nullable=True, comment="이름 변경 마커")

    # 주식 수량 정보
    NumNewShares: Optional[float] = Column(Float, nullable=True, comment="신주 수량")
    NumOldShares: Optional[float] = Column(Float, nullable=True, comment="구주 수량")

    # 통화 및 금액 정보
    ISOCurrCode: Optional[str] = Column(String(3), nullable=True, comment="ISO 통화 코드")
    CashAmt: Optional[float] = Column(Float, nullable=True, comment="현금 금액")

    # 발행 마커
    MultiIssueMarker: Optional[str] = Column(String(1), nullable=True, comment="복수 발행 마커")
    CmplxIssueMarker: Optional[str] = Column(String(1), nullable=True, comment="복잡 발행 마커")
    OfferCmpyName: Optional[str] = Column(String(91), nullable=True, comment="제안 회사명")

    # 주요 일자
    AnnouncedDate: Optional[datetime] = Column(DateTime, nullable=True, comment="공시일")
    RecordDate: Optional[datetime] = Column(DateTime, nullable=True, comment="기준일")
    EffectiveDate: Optional[datetime] = Column(DateTime, nullable=True, comment="효력 발생일")
    ExpiryDate: Optional[datetime] = Column(DateTime, nullable=True, comment="만료일")

    # 기타
    UnmatchDsCode: Optional[str] = Column(String(13), nullable=True, comment="미매칭 DS 코드")
    LicFlag: Optional[int] = Column(SmallInteger, nullable=True, comment="라이선스 비트맵 플래그")

    def __repr__(self) -> str:
        return (
            f"<DS2CapEvent(InfoCode={self.InfoCode}, "
            f"EventNum={self.EventNum}, "
            f"ActionTypeCode={self.ActionTypeCode})>"
        )


class Ds2MnemChg(Base):
    """
    DS2 Mnemonic Change 테이블
    
    종목 니모닉/티커 변경 이력을 저장하는 테이블
    """
    __tablename__ = "Ds2MnemChg"
    __table_args__ = (
        PrimaryKeyConstraint("InfoCode", "StartDate", name="pkey_Ds2MnemChg"),
        Index("Ds2MnemChg_1", "DsMnem", "StartDate", "InfoCode"),
        {"schema": "dbo"},
    )

    # Primary Key
    InfoCode: int = Column(Integer, nullable=False, comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)")
    StartDate: datetime = Column(DateTime, nullable=False, comment="시작일")

    # 변경 정보
    EndDate: Optional[datetime] = Column(DateTime, nullable=True, comment="종료일")
    DsMnem: Optional[str] = Column(String(13), nullable=True, comment="Datastream 니모닉")
    Ticker: Optional[str] = Column(String(13), nullable=True, comment="티커")

    def __repr__(self) -> str:
        return (
            f"<Ds2MnemChg(InfoCode={self.InfoCode}, "
            f"Ticker={self.Ticker}, "
            f"StartDate={self.StartDate})>"
        )


class DS2Exchange(Base):
    """
    DS2 Exchange 테이블
    
    거래소 정보를 저장하는 마스터 테이블
    """
    __tablename__ = "DS2Exchange"
    __table_args__ = (
        PrimaryKeyConstraint("ExchIntCode", name="pkey_DS2Exchange"),
        Index("DS2Exchange_1", "DsExchCode"),
        Index("DS2Exchange_2", "ExchMnem"),
        {"schema": "dbo"},
    )

    # Primary Key
    ExchIntCode: int = Column(SmallInteger, nullable=False, comment="거래소 내부 코드")

    # 거래소 코드 정보
    DsExchCode: Optional[str] = Column(String(2), nullable=True, comment="Datastream 거래소 코드")
    ExchType: Optional[str] = Column(String(4), nullable=True, comment="거래소 유형")
    ExchName: Optional[str] = Column(String(91), nullable=True, comment="거래소명")
    ExchMnem: Optional[str] = Column(String(3), nullable=True, comment="거래소 니모닉")

    # 국가 정보
    ExchCtryCode: Optional[str] = Column(String(7), nullable=True, comment="거래소 국가 코드")
    CtryCodeType: Optional[int] = Column(SmallInteger, nullable=True, comment="국가 코드 유형")

    def __repr__(self) -> str:
        return (
            f"<DS2Exchange(ExchIntCode={self.ExchIntCode}, "
            f"ExchName={self.ExchName})>"
        )


class DS2Region(Base):
    """
    DS2 Region 테이블
    
    지역/국가 정보를 저장하는 마스터 테이블
    """
    __tablename__ = "DS2Region"
    __table_args__ = (
        PrimaryKeyConstraint("Region", "RegCodeTypeID", name="pkey_DS2Region"),
        Index("DS2Region_1", "RegCodeTypeID"),
        Index("DS2Region_2", "Name_"),
        {"schema": "dbo"},
    )

    # Primary Key
    Region: str = Column(String(7), nullable=False, comment="지역 코드")
    RegCodeTypeID: int = Column(SmallInteger, nullable=False, comment="지역 코드 유형 ID")

    # 지역 정보
    Name_: Optional[str] = Column("Name_", String(91), nullable=True, comment="지역명")
    DsGeoCode: Optional[str] = Column(String(2), nullable=True, comment="Datastream 지역 코드")
    ISOCurrCode: Optional[str] = Column(String(3), nullable=True, comment="ISO 통화 코드")
    PermRegion: Optional[str] = Column(String(16), nullable=True, comment="영구 지역 코드")

    def __repr__(self) -> str:
        return (
            f"<DS2Region(Region={self.Region}, "
            f"Name_={self.Name_})>"
        )


class RKDFndInfo(Base):
    """
    RKD Fund Info 테이블
    
    펀드/종목 기본 정보 테이블
    - 티커, ISIN, CUSIP, SEDOL 등 다양한 식별자 포함
    """
    __tablename__ = "RKDFndInfo"
    __table_args__ = (
        PrimaryKeyConstraint("Code", name="pkey_RKDFNDINFO"),
        {"schema": "dbo"},
    )

    # Primary Key
    Code: int = Column(Integer, nullable=False, comment="코드")

    # 식별자 정보
    RepNo: Optional[str] = Column(String(5), nullable=True, comment="리포트 번호")
    Name_: Optional[str] = Column("Name_", String(60), nullable=True, comment="종목명")
    Ticker: Optional[str] = Column(String(40), nullable=True, comment="티커")
    Cusip: Optional[str] = Column(String(8), nullable=True, comment="CUSIP 코드")
    ISIN: Optional[str] = Column(String(40), nullable=True, comment="ISIN 코드")
    RIC: Optional[str] = Column(String(40), nullable=True, comment="RIC 코드")
    Sedol: Optional[str] = Column(String(6), nullable=True, comment="SEDOL 코드")

    # 거래소 및 지역 정보
    ExchCode: Optional[str] = Column(String(4), nullable=True, comment="거래소 코드")
    RegionCode: Optional[str] = Column(String(40), nullable=True, comment="지역 코드")
    CntryCode: Optional[str] = Column(String(3), nullable=True, comment="국가 코드")

    def __repr__(self) -> str:
        return (
            f"<RKDFndInfo(Code={self.Code}, "
            f"Name_={self.Name_}, "
            f"Ticker={self.Ticker})>"
        )


class RKDFndCmpIssDet(Base):
    """
    RKD Fund Company Issue Detail 테이블
    
    회사 발행 주식 상세 정보 테이블
    - 발행주식수, 유통주식수, 자사주 등
    """
    __tablename__ = "RKDFndCmpIssDet"
    __table_args__ = (
        PrimaryKeyConstraint("Code", "IssueID", name="pkey_RKDFndCmpIssDet"),
        {"schema": "dbo"},
    )

    # Primary Key
    Code: int = Column(Integer, nullable=False, comment="코드")
    IssueID: int = Column(Integer, nullable=False, comment="발행 ID")

    # 발행 유형 정보
    IssueTypeCode: Optional[int] = Column(Integer, nullable=True, comment="발행 유형 코드")
    IssueOrder: Optional[int] = Column(Integer, nullable=True, comment="발행 순서")

    # 액면가 정보
    ISOParCurrCode: Optional[str] = Column(String(4), nullable=True, comment="액면가 통화 코드")
    ParValue: Optional[float] = Column(Float, nullable=True, comment="액면가")

    # 주식 수량 정보
    ShAuth: Optional[float] = Column(Float, nullable=True, comment="수권주식수")
    ShOut: Optional[float] = Column(Float, nullable=True, comment="발행주식수")
    ShOutDt: Optional[datetime] = Column(DateTime, nullable=True, comment="발행주식수 기준일")
    IssueFloat: Optional[float] = Column(Float, nullable=True, comment="유통주식수")
    FloatDt: Optional[datetime] = Column(DateTime, nullable=True, comment="유통주식수 기준일")
    ShIssued: Optional[float] = Column(Float, nullable=True, comment="발행된 주식수")
    ShIssuedDt: Optional[datetime] = Column(DateTime, nullable=True, comment="발행된 주식수 기준일")
    ShAuthDt: Optional[datetime] = Column(DateTime, nullable=True, comment="수권주식수 기준일")

    # 자사주 정보
    TreasurySh: Optional[float] = Column(Float, nullable=True, comment="자사주")
    TreasuryShDt: Optional[datetime] = Column(DateTime, nullable=True, comment="자사주 기준일")

    # 기타
    Votes: Optional[float] = Column(Float, nullable=True, comment="의결권")
    ConvFactor: Optional[float] = Column(Float, nullable=True, comment="전환 비율")
    LicFlag: Optional[int] = Column(Integer, nullable=True, comment="라이선스 플래그")

    def __repr__(self) -> str:
        return (
            f"<RKDFndCmpIssDet(Code={self.Code}, "
            f"IssueID={self.IssueID})>"
        )


class DS2Company(Base):
    """
    DS2 Company 테이블
    
    회사 정보 마스터 테이블
    """
    __tablename__ = "DS2Company"
    __table_args__ = (
        PrimaryKeyConstraint("DsCmpyCode", name="pkey_DS2Company"),
        Index("DS2Company_1", "CmpyCtryCode"),
        {"schema": "dbo"},
    )

    # Primary Key
    DsCmpyCode: int = Column(Integer, nullable=False, comment="Datastream 회사 코드")

    # 회사 정보
    DsCompCode: Optional[str] = Column(String(13), nullable=True, comment="Datastream 회사 코드 (문자)")
    DsCmpyName: Optional[str] = Column(String(91), nullable=True, comment="회사명")

    # 국가 정보
    CmpyCtryCode: Optional[str] = Column(String(7), nullable=True, comment="회사 국가 코드")
    CmpyCtryType: Optional[int] = Column(SmallInteger, nullable=True, comment="국가 코드 유형")

    # 산업 정보
    IndusIsDef: Optional[str] = Column(String(2), nullable=True, comment="산업 정의 여부")

    def __repr__(self) -> str:
        return (
            f"<DS2Company(DsCmpyCode={self.DsCmpyCode}, "
            f"DsCmpyName={self.DsCmpyName})>"
        )


class DS2Security(Base):
    """
    DS2 Security 테이블
    
    증권 정보 마스터 테이블
    - 회사(DS2Company)와 종목(DS2CtryQtInfo)을 연결
    """
    __tablename__ = "DS2Security"
    __table_args__ = (
        PrimaryKeyConstraint("DsSecCode", name="pkey_DS2Security"),
        Index("DS2Security_1", "DsCmpyCode", "IsMajorSec"),
        {"schema": "dbo"},
    )

    # Primary Key
    DsSecCode: int = Column(Integer, nullable=False, comment="Datastream 증권 코드")

    # 증권 코드 정보
    DsSctyCode: Optional[str] = Column(String(13), nullable=True, comment="Datastream 증권 코드 (문자)")
    DsCmpyCode: Optional[int] = Column(Integer, nullable=True, comment="Datastream 회사 코드 (DS2Company 참조)")
    IsMajorSec: Optional[str] = Column(String(2), nullable=True, comment="주요 증권 여부")

    # 증권 정보
    DsSecName: Optional[str] = Column(String(91), nullable=True, comment="증권명")
    ISOCurrCode: Optional[str] = Column(String(3), nullable=True, comment="ISO 통화 코드")
    DivUnit: Optional[str] = Column(String(7), nullable=True, comment="배당 단위")

    # 주요 시세 정보
    PrimQtSedol: Optional[str] = Column(String(7), nullable=True, comment="주요 시세 SEDOL")
    PrimExchMnem: Optional[str] = Column(String(3), nullable=True, comment="주요 거래소 니모닉")
    PrimQtInfoCode: Optional[int] = Column(Integer, nullable=True, comment="주요 시세 InfoCode (DS2CtryQtInfo 참조)")

    # 외부 식별자
    WSSctyPPI: Optional[str] = Column(String(13), nullable=True, comment="WorldScope PPI")
    IBESTicker: Optional[str] = Column(String(13), nullable=True, comment="IBES 티커")
    WSSctyPPI2: Optional[str] = Column(String(13), nullable=True, comment="WorldScope PPI 2")
    IBESTicker2: Optional[str] = Column(String(13), nullable=True, comment="IBES 티커 2")

    # 상장폐지 정보
    DelistDate: Optional[datetime] = Column(DateTime, nullable=True, comment="상장폐지일")

    def __repr__(self) -> str:
        return (
            f"<DS2Security(DsSecCode={self.DsSecCode}, "
            f"DsSecName={self.DsSecName})>"
        )


class DS2EquityIndex(Base):
    """
    DS2 Equity Index 테이블
    
    주가 지수 마스터 테이블
    - KOSPI, S&P 500 등 각종 지수 정보
    """
    __tablename__ = "DS2EquityIndex"
    __table_args__ = (
        PrimaryKeyConstraint("DSIndexCode", name="pkey_DS2EquityIndex"),
        Index("DS2EquityIndex_1", "DSIndexMnem"),
        Index("DS2EquityIndex_2", "ISOCurrCode"),
        Index("DS2EquityIndex_3", "IndexDesc"),
        Index("DS2EquityIndex_4", "Region"),
        {"schema": "dbo"},
    )

    # Primary Key
    DSIndexCode: int = Column(Integer, nullable=False, comment="Datastream 인덱스 코드")

    # 인덱스 식별 정보
    DSIndexMnem: Optional[str] = Column(String(13), nullable=True, comment="인덱스 니모닉")
    IndexDesc: Optional[str] = Column(String(255), nullable=True, comment="인덱스 설명")

    # 지역 정보
    Region: Optional[str] = Column(String(7), nullable=True, comment="지역 코드")
    RegCodeTypeID: Optional[int] = Column(SmallInteger, nullable=True, comment="지역 코드 유형 ID")

    # 기타 정보
    LDB: Optional[str] = Column(String(3), nullable=True, comment="LDB 코드")
    SourceCode: Optional[str] = Column(String(5), nullable=True, comment="소스 코드")
    IndexListCode: Optional[int] = Column(Integer, nullable=True, comment="인덱스 목록 코드")
    BaseDate: Optional[datetime] = Column(DateTime, nullable=True, comment="기준일")

    # 통화 정보
    ISOCurrCode: Optional[str] = Column(String(3), nullable=True, comment="ISO 통화 코드")
    IsLocalCurrency: Optional[str] = Column(String(1), nullable=True, comment="로컬 통화 여부")

    # 상태 정보
    IndexStatusCode: Optional[str] = Column(String(1), nullable=True, comment="인덱스 상태 코드")
    LicFlag: Optional[int] = Column(SmallInteger, nullable=True, comment="라이선스 비트맵 플래그")

    def __repr__(self) -> str:
        return (
            f"<DS2EquityIndex(DSIndexCode={self.DSIndexCode}, "
            f"IndexDesc={self.IndexDesc})>"
        )


class DS2IndexData(Base):
    """
    DS2 Index Data 테이블
    
    주가 지수 일별 데이터 테이블
    - PI: Price Index (가격 지수)
    - RI: Return Index (수익률 지수)
    - MV: Market Value (시가총액)
    """
    __tablename__ = "DS2IndexData"
    __table_args__ = (
        PrimaryKeyConstraint("DSIndexCode", "ValueDate", name="pkey_DS2IndexData"),
        {"schema": "dbo"},
    )

    # Primary Key
    DSIndexCode: int = Column(Integer, nullable=False, comment="Datastream 인덱스 코드 (DS2EquityIndex 참조)")
    ValueDate: datetime = Column(DateTime, nullable=False, comment="기준일")

    # 지수 데이터
    PI_: Optional[float] = Column("PI_", Float, nullable=True, comment="가격 지수 (Price Index)")
    RI: Optional[float] = Column(Float, nullable=True, comment="수익률 지수 (Return Index)")
    MV: Optional[float] = Column(Float, nullable=True, comment="시가총액 (Market Value)")

    # 메타 정보
    LicFlag: Optional[int] = Column(SmallInteger, nullable=True, comment="라이선스 비트맵 플래그")

    def __repr__(self) -> str:
        return (
            f"<DS2IndexData(DSIndexCode={self.DSIndexCode}, "
            f"ValueDate={self.ValueDate}, "
            f"PI_={self.PI_})>"
        )


# =============================================================================
# 뷰 (Views) - 읽기 전용
# =============================================================================


class VwDs2MktCap(Base):
    """
    DS2 Market Cap 뷰 (읽기 전용)
    
    시가총액 정보를 제공하는 뷰
    - MktCap = NumShrs * UnadjClose
    """
    __tablename__ = "vw_Ds2MktCap"
    __table_args__ = {"schema": "dbo"}

    # Primary Key (Unique Key: InfoCode, MarketDate)
    InfoCode: int = Column(
        Integer, primary_key=True, 
        comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)"
    )
    MarketDate: datetime = Column(
        DateTime, primary_key=True, 
        comment="거래일"
    )

    # 거래소 정보
    PrimExchIntCode: int = Column(
        SmallInteger, nullable=False, 
        comment="주요 거래소 코드"
    )
    PrimaryExchange: Optional[str] = Column(
        String(91), nullable=True, 
        comment="주요 거래소명"
    )

    # 가격 및 주식수
    UnadjClose: Optional[float] = Column(
        Float, nullable=True, 
        comment="미수정 종가"
    )
    NumShrs: Optional[float] = Column(
        Float, nullable=True, 
        comment="미수정 발행주식수 (원본 * 1000)"
    )

    # 시가총액
    MktCap: Optional[float] = Column(
        Float, nullable=True, 
        comment="시가총액 (NumShrs * UnadjClose)"
    )

    # 통화 및 이벤트
    Currency: Optional[str] = Column(
        String(3), nullable=True, 
        comment="ISO 통화 코드"
    )
    EventDate: Optional[datetime] = Column(
        DateTime, nullable=True, 
        comment="발행주식수 입력일"
    )

    def __repr__(self) -> str:
        return (
            f"<VwDs2MktCap(InfoCode={self.InfoCode}, "
            f"MarketDate={self.MarketDate}, "
            f"MktCap={self.MktCap})>"
        )


class VwDs2Pricing(Base):
    """
    DS2 Pricing 뷰 (읽기 전용)
    
    모든 거래소의 가격 데이터를 조정 유형(AdjType)에 따라 제공하는 뷰
    - AdjType: 0=미수정, 1=분할만 조정, 2=전체 조정
    - CumAdjFactor: 누적 조정 계수 (최신=1, 과거로 갈수록 곱해짐)
    """
    __tablename__ = "vw_Ds2Pricing"
    __table_args__ = {"schema": "dbo"}

    # Primary Key (복합키로 유니크 식별)
    InfoCode: int = Column(
        Integer, primary_key=True,
        comment="Datastream 내부 코드 (DS2CtryQtInfo 참조)"
    )
    MarketDate: datetime = Column(
        DateTime, primary_key=True,
        comment="거래일"
    )
    AdjType: int = Column(
        Integer, primary_key=True,
        comment="조정 유형 (0=미수정, 1=분할만, 2=전체)"
    )
    ExchIntCode: Optional[int] = Column(
        SmallInteger, primary_key=True,
        comment="거래소 코드"
    )

    # 종목 코드
    DsCode: Optional[str] = Column(
        String(13), nullable=True,
        comment="Datastream 코드"
    )

    # 거래소 정보
    IsPrimExchQt: Optional[str] = Column(
        String(1), nullable=True,
        comment="주요 거래소 시세 여부 (Y/N)"
    )
    ExchName: Optional[str] = Column(
        String(91), nullable=True,
        comment="거래소명"
    )

    # 조정 계수
    CumAdjFactor: Optional[float] = Column(
        Float, nullable=True,
        comment="누적 조정 계수 (최신=1, 역순 누적)"
    )
    PriceUnitAdjustment: Optional[str] = Column(
        String(10), nullable=True,
        comment="가격 단위 조정 (E+00=1:1, E-02=1/100)"
    )

    # OHLCV 데이터
    Open_: Optional[float] = Column(
        "Open_", Float, nullable=True,
        comment="시가"
    )
    High: Optional[float] = Column(
        Float, nullable=True,
        comment="고가"
    )
    Low: Optional[float] = Column(
        Float, nullable=True,
        comment="저가"
    )
    Close_: Optional[float] = Column(
        "Close_", Float, nullable=True,
        comment="종가"
    )
    Volume: Optional[float] = Column(
        Float, nullable=True,
        comment="거래량"
    )

    # 호가 데이터 (장마감 시점)
    Bid: Optional[float] = Column(
        Float, nullable=True,
        comment="장마감 시 매수 호가"
    )
    Ask: Optional[float] = Column(
        Float, nullable=True,
        comment="장마감 시 매도 호가"
    )

    # 추가 가격 데이터
    VWAP: Optional[float] = Column(
        Float, nullable=True,
        comment="거래량가중평균가격"
    )
    ConsolVol: Optional[float] = Column(
        Float, nullable=True,
        comment="통합 거래량"
    )

    # 통화
    Currency: Optional[str] = Column(
        String(3), nullable=True,
        comment="ISO 통화 코드"
    )

    def __repr__(self) -> str:
        return (
            f"<VwDs2Pricing(InfoCode={self.InfoCode}, "
            f"MarketDate={self.MarketDate}, "
            f"AdjType={self.AdjType}, "
            f"Close_={self.Close_})>"
        )
