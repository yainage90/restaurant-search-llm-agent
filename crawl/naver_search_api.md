1. URL
https://openapi.naver.com/v1/search/local

2. request
get /v1/search/local
HOST: openapi.naver.com
Content-Type: plain/text
X-Naver-Client-Id: <env: NAVER_CLIENT_ID>
X-Naver-Client-Secret: <env: NAVER_CLIENT_SECRET>

3. request params
{
    "query": "강남역 국밥",
    "display": 5,
    "start": 1,
    "sort": "comment"
}

4. response

```json
{
  "lastBuildDate": "Thu, 31 Jul 2025 22:46:25 +0900",
  "total": 5,
  "start": 1,
  "display": 5,
  "items": [
    {
      "title": "농민백암순대",
      "link": "",
      "category": "한식>순대,순댓국",
      "description": "",
      "telephone": "",
      "address": "서울특별시 강남구 역삼동 830-9",
      "roadAddress": "서울특별시 강남구 역삼로3길 20-4",
      "mapx": "1270314467",
      "mapy": "374949366"
    },
    {
      "title": "베트남이랑",
      "link": "",
      "category": "음식점>베트남음식",
      "description": "",
      "telephone": "",
      "address": "서울특별시 서초구 서초동 1317-5 대경빌딩 지하 1층",
      "roadAddress": "서울특별시 서초구 서초대로77길 15 대경빌딩 지하 1층",
      "mapx": "1270261395",
      "mapy": "374989957"
    },
    {
      "title": "강남진해장",
      "link": "",
      "category": "음식점>한식",
      "description": "",
      "telephone": "",
      "address": "서울특별시 강남구 역삼동 819-4",
      "roadAddress": "서울특별시 강남구 테헤란로5길 11",
      "mapx": "1270292442",
      "mapy": "374995538"
    },
    {
      "title": "화설 강남직영점",
      "link": "https://app.catchtable.co.kr/ct/shop/hwaseol.g",
      "category": "한식>육류,고기요리",
      "description": "",
      "telephone": "",
      "address": "서울특별시 서초구 서초동 1308-24 1층",
      "roadAddress": "서울특별시 서초구 서초대로73길 38 1층",
      "mapx": "1270245770",
      "mapy": "375006909"
    },
    {
      "title": "보승회관 <b>강남역</b>직영점",
      "link": "https://www.boseunghall.com/",
      "category": "음식점>한식>국밥",
      "description": "",
      "telephone": "",
      "address": "서울특별시 강남구 역삼동 820-2 101호",
      "roadAddress": "서울특별시 강남구 테헤란로1길 17 101호",
      "mapx": "1270276883",
      "mapy": "374994834"
    }
  ]
}
```
