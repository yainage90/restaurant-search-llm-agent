
import json
import re
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import quote
import os

@dataclass
class CrawlerConfig:
    base_url: str = "https://map.naver.com/p/search/"
    default_wait_time: int = 1.5
    max_review_clicks: int = 10
    
    # CSS Selectors
    search_iframe_id: str = "searchIframe"
    entry_iframe_id: str = "entryIframe"
    first_result_selector: str = "li.UEzoS:first-child .place_bluelink"
    tab_container_selector: str = ".Jxtsc"
    menu_more_button_selector: str = ".fvwqf"
    # Multiple menu item selectors for different layouts
    menu_item_selectors: List[str] = None
    menu_name_selectors: List[str] = None
    menu_price_selectors: List[str] = None
    # Sub-tab selectors (for menu category tabs etc.)
    review_more_button_selector: str = "a.fvwqf"
    review_container_selector: str = "ul#_review_list"
    review_item_selector: str = "div.pui__vn15t2 > a"
    # Restaurant info selectors
    info_description_selectors: List[str] = None
    
    def __post_init__(self):
        # Define multiple selector patterns for different menu layouts
        self.menu_item_selectors = [
            "div.MenuContent__info_detail__rCviz",  # For Starbucks-like infinite scroll menus
            "div.MXkFw",  # Original selector for standard layout
            # "li[class*='item']",  # Items with class containing 'item'
            # ".place_section_content li",  # Alternative for sub-tab layout
            # "div[class*='menu'] li",  # Menu container lists
            # "ul li",  # Generic fallback - but will validate content
        ]
        self.menu_name_selectors = [
            "div.MenuContent__tit__313LA", # For Starbucks-like infinite scroll menus
            ".lPzHi",  # Original selector
            # ".name, .item_name, .menu_name",  # Alternative selectors
            # "strong",  # Common name patterns
            # ".title, .label",  # Title/label patterns
            # "span:first-child",  # Fallback pattern
        ]
        self.menu_price_selectors = [
            "div.MenuContent__price__lhCy9", # For Starbucks-like infinite scroll menus
            ".GXS1X",  # Original selector
            # ".price, .item_price, .menu_price",  # Alternative selectors
            # ".cost, .amount",  # Cost patterns
            # "span:last-child",  # Fallback pattern
        ]
        self.info_description_selectors = [
            "div.T8RFa",  # New requirement: specific selector for description
        ]

# 설정 및 경로
config = CrawlerConfig()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data/restaurants.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "../data/crawled_restaurants.jsonl")

log_dir = f'{BASE_DIR}/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def get_driver() -> webdriver.Chrome:
    """WebDriver를 설정하고 반환합니다."""
    logger.info("WebDriver 초기화 중...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    logger.info("WebDriver 초기화 완료")
    return driver

def wait_for_element_clickable(driver: webdriver.Chrome, selector: str, timeout: int = 5) -> bool:
    """요소가 클릭 가능할 때까지 효율적으로 대기합니다."""
    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        return True
    except TimeoutException:
        return False

def switch_to_entry_iframe(driver: webdriver.Chrome, wait: WebDriverWait) -> bool:
    try:
        driver.switch_to.default_content()
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, config.entry_iframe_id)))
        logger.info("단일 상세 페이지 Iframe으로 전환 성공")
        return True
    except TimeoutException:
        # 단일 상세 페이지를 불러오지 못했을 경우 실패
        return False


def click_tab(driver: webdriver.Chrome, wait: WebDriverWait, tab_name: str) -> bool:
    """지정된 탭을 클릭합니다."""
    try:
        # 먼저 사용 가능한 모든 탭을 로깅
        try:
            tab_container = driver.find_element(By.CSS_SELECTOR, ".Jxtsc")
            all_tabs = tab_container.find_elements(By.TAG_NAME, "a")
            available_tabs = [tab.text.strip() for tab in all_tabs if tab.text.strip()]
            logger.info(f"사용 가능한 탭들: {available_tabs}")
        except:
            logger.warning("탭 목록을 가져올 수 없습니다.")
        
        # 다양한 탭 이름 패턴 시도
        tab_patterns = [
            f"//div[contains(@class, 'Jxtsc')]//a[.//span[text()='{tab_name}']]",
            # f"//div[contains(@class, 'Jxtsc')]//a[contains(text(), '{tab_name}')]",
            f"//a[.//span[text()='{tab_name}']]",
            # f"//a[contains(text(), '{tab_name}')]"
        ]
        
        for pattern in tab_patterns:
            try:
                tab = wait.until(EC.element_to_be_clickable((By.XPATH, pattern)))
                logger.info(f"'{tab_name}' 탭을 찾았습니다. pattern: {pattern}")
                driver.execute_script("arguments[0].click();", tab)
                # 탭 전환 후 콘텐츠 로딩 대기
                WebDriverWait(driver, config.default_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
                )
                return True
            except TimeoutException:
                continue
        
        logger.warning(f"'{tab_name}' 탭을 찾을 수 없습니다.")
        return False
    except Exception as e:
        logger.error(f"탭 클릭 중 오류 발생: {e}")
        return False

def click_more_button_until_done(driver: webdriver.Chrome, selector: str, max_clicks: int = None) -> int:
    """더보기 버튼을 찾을 수 없을 때까지 계속 클릭합니다."""
    click_count = 0
    max_clicks = max_clicks or float('inf')
    
    # 더 다양한 더보기 버튼 셀렉터 시도
    more_button_selectors = [
        selector,  # 기본 셀렉터
        "a.fvwqf",
        # "button.fvwqf", 
        # ".more_btn",
        # "[class*='more']",
        # "a[contains(text(), '더보기')]",
        # "button[contains(text(), '더보기')]",
        # ".btn_more",
        # ".expand",
        # "[data-action='expand']"
    ]
    
    while click_count < max_clicks:
        button_found = False
        
        for btn_selector in more_button_selectors:
            try:
                more_button = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, btn_selector))
                )
                logger.debug(f"더보기 버튼 발견: {btn_selector}")
                driver.execute_script("arguments[0].click();", more_button)
                click_count += 1
                button_found = True
                
                # 효율적인 대기
                import time
                time.sleep(config.default_wait_time)
                break
                
            except TimeoutException:
                continue
            except Exception as e:
                logger.debug(f"더보기 버튼 클릭 중 예외 ({btn_selector}): {e}")
                continue
        
        if not button_found:
            break
    
    if click_count > 0:
        logger.info(f"총 {click_count}번의 더보기 버튼을 클릭했습니다.")
    
    return click_count

def validate_menu_item(name: str, price: str) -> bool:
    """메뉴 아이템의 유효성을 검증합니다."""
    if not (name and name.strip() and price and price.strip()):
        return False
    
    name_clean = name.strip()
    price_clean = price.strip()
    
    # 메뉴 이름과 가격이 같으면 카테고리일 가능성이 높음
    if name_clean == price_clean:
        return False
    
    return True

def try_menu_selectors(driver: webdriver.Chrome, wait: WebDriverWait) -> Optional[tuple]:
    """다양한 CSS 셀렉터를 시도하여 메뉴 항목을 찾습니다."""
    for item_selector in config.menu_item_selectors:
        try:
            menu_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, item_selector)))
            if menu_elements:
                # 해당 셀렉터에 맞는 name/price 셀렉터 찾기
                for name_selector in config.menu_name_selectors:
                    for price_selector in config.menu_price_selectors:
                        try:
                            # 첫 번째 요소로 테스트
                            test_el = menu_elements[0]
                            name = test_el.find_element(By.CSS_SELECTOR, name_selector).text
                            price = test_el.find_element(By.CSS_SELECTOR, price_selector).text
                            if validate_menu_item(name, price):
                                logger.info(f"메뉴 셀렉터 찾음: items={item_selector}, name={name_selector}, price={price_selector}")
                                return (menu_elements, name_selector, price_selector)
                        except (NoSuchElementException, TimeoutException):
                            continue
        except TimeoutException:
            continue
    return None

def try_price_based_extraction(driver: webdriver.Chrome) -> List[Dict[str, str]]:
    """가격 기반으로 메뉴 항목을 추출합니다. (스타벅스 등 특수 구조용)"""
    import re
    
    try:
        # 가격이 포함된 모든 요소 찾기 (더 다양한 패턴)
        price_patterns = [
            "//*[contains(text(), '원')]",
            "//*[contains(text(), '₩')]", 
            "//*[contains(text(), ',')]",  # 쉼표가 있는 숫자
            "//*[text()[matches(., '\\d{3,}')]]"  # 3자리 이상 숫자
        ]
        
        all_price_elements = []
        for pattern in price_patterns:
            try:
                elements = driver.find_elements(By.XPATH, pattern)
                all_price_elements.extend(elements)
            except:
                continue
        
        logger.info(f"가격 포함 요소 {len(all_price_elements)}개 발견")
        
        valid_menu_items = []
        processed_prices = set()
        processed_names = set()
        
        for elem in all_price_elements:
            try:
                price_text = elem.text.strip()
                
                # 더 관대한 가격 형식 검증
                price_patterns = [
                    r'^\d{1,2},?\d{3}원$',  # 6,500원
                    r'^\d{4,5}원$',         # 6500원
                    r'^\d{1,2},?\d{3}$',    # 6,500
                    r'^₩\d{1,2},?\d{3}$'    # ₩6,500
                ]
                
                is_valid_price = any(re.match(pattern, price_text) for pattern in price_patterns)
                if not is_valid_price or price_text in processed_prices:
                    continue
                
                processed_prices.add(price_text)
                
                # 가격 요소의 조상 요소들에서 메뉴 이름 찾기 (더 넓은 범위)
                current = elem
                found_menu = False
                
                for level in range(8):  # 최대 8단계까지 올라가며 검색
                    try:
                        current = current.find_element(By.XPATH, "..")
                        
                        # 형제 요소들과 자식 요소들 모두 검색
                        search_elements = []
                        search_elements.extend(current.find_elements(By.XPATH, ".//*"))
                        search_elements.extend(current.find_elements(By.XPATH, "..//*"))
                        
                        for search_elem in search_elements:
                            sibling_text = search_elem.text.strip()
                            
                            # 메뉴명처럼 보이는 텍스트 찾기 (더 관대한 조건)
                            if (sibling_text and 
                                price_text not in sibling_text and
                                len(sibling_text) >= 2 and 
                                len(sibling_text) <= 150 and
                                '원' not in sibling_text and
                                not re.search(r'\d{3,}', sibling_text) and  # 큰 숫자 없음
                                sibling_text not in processed_names):
                                
                                # 더 관대한 카테고리 키워드 체크
                                category_keywords = [
                                    '음료', '추천', '에스프레소 / 커피', '디카페인 커피', '블론드 커피', 
                                    '콜드 브루', '프라푸치노 / 블렌디드', '티 음료', '피지오 / 리프레셔',
                                    '기타 (딸기 라떼, 초콜릿 음료)', '병음료', '메뉴', '카테고리',
                                    '전체보기', '상세보기', '이미지', '사진', '리뷰', '별점'
                                ]
                                
                                if not any(keyword in sibling_text for keyword in category_keywords):
                                    menu_item = {
                                        "name": sibling_text.strip(), 
                                        "price": price_text
                                    }
                                    valid_menu_items.append(menu_item)
                                    processed_names.add(sibling_text)
                                    found_menu = True
                                    logger.debug(f"메뉴 발견: {sibling_text} - {price_text}")
                                    break
                        
                        if found_menu:
                            break
                            
                    except:
                        break
                        
            except:
                continue
        
        # 중복 제거
        unique_menus = []
        seen_names = set()
        for menu in valid_menu_items:
            if menu['name'] not in seen_names:
                unique_menus.append(menu)
                seen_names.add(menu['name'])
        
        logger.info(f"가격 기반 추출로 {len(unique_menus)}개 메뉴 발견")
        return unique_menus
        
    except Exception as e:
        logger.warning(f"가격 기반 추출 중 오류: {e}")
        return []


def scroll_until_no_new_content(driver: webdriver.Chrome, wait: WebDriverWait, item_selector: str) -> int:
    """무한 스크롤 페이지에서 새로운 콘텐츠가 없을 때까지 스크롤합니다."""
    scroll_count = 0
    no_new_content_strikes = 0
    
    logger.info(f"무한 스크롤 시작 (아이템 셀렉터: {item_selector})")

    while no_new_content_strikes < 3:  # 3번 연속으로 새 아이템이 없으면 중단
        try:
            menu_elements = driver.find_elements(By.CSS_SELECTOR, item_selector)
            initial_item_count = len(menu_elements)
            
            if not menu_elements:
                logger.warning("스크롤할 아이템을 찾지 못했습니다.")
                break

            # 마지막 요소를 뷰로 스크롤하여 새 콘텐츠 로드를 유발
            last_element = menu_elements[-1]
            driver.execute_script("arguments[0].scrollIntoView(true);", last_element)
            scroll_count += 1
            
            # 콘텐츠 로딩 대기 (네트워크 상태에 따라 조절)
            import time
            time.sleep(config.default_wait_time)

            final_item_count = len(driver.find_elements(By.CSS_SELECTOR, item_selector))
            
            if final_item_count > initial_item_count:
                logger.info(f"스크롤 {scroll_count}: {final_item_count - initial_item_count}개 새 아이템 발견 (총 {final_item_count}개)")
                no_new_content_strikes = 0
            else:
                logger.info(f"스크롤 {scroll_count}: 새 아이템 없음 (총 {final_item_count}개)")
                no_new_content_strikes += 1
        except Exception as e:
            logger.warning(f"스크롤 중 예외 발생: {e}")
            break
            
    logger.info(f"{scroll_count}번 스크롤 후 종료.")
    return scroll_count

def get_menu_data(driver: webdriver.Chrome, wait: WebDriverWait) -> Optional[List[Dict[str, str]]]:
    """메뉴 탭의 모든 메뉴명과 가격을 수집합니다."""
    if not click_tab(driver, wait, "메뉴"):
        return None

    logger.info("메뉴 정보 수집 시작")

    # 서브탭을 클릭하지 않고 현재 선택된 탭의 메뉴만 수집
    logger.info("현재 선택된 메뉴 탭에서 메뉴를 수집합니다.")
    
    # "더보기" 버튼이 모두 사라질 때까지 클릭
    click_count = click_more_button_until_done(driver, config.menu_more_button_selector, max_clicks=20)
    if click_count > 0:
        logger.info(f"메뉴 더보기를 {click_count}번 클릭했습니다.")
        import time
        time.sleep(config.default_wait_time)

    # 무한 스크롤 처리 (e.g., 스타벅스)
    # "더보기" 버튼이 없는 무한 스크롤 방식의 페이지에 대응합니다.
    try:
        # 스크롤이 필요한 특정 메뉴 구조인지 확인 (스타벅스 케이스)
        infinite_scroll_selector = "div.MenuContent__info_detail__rCviz"
        if driver.find_elements(By.CSS_SELECTOR, infinite_scroll_selector):
            logger.info("무한 스크롤 메뉴 구조 감지. 스크롤을 시작합니다.")
            scroll_until_no_new_content(driver, wait, infinite_scroll_selector)
            # 스크롤 후 잠시 대기하여 DOM 안정화
            import time
            time.sleep(1)
    except Exception as e:
        logger.warning(f"무한 스크롤 확인 중 오류 발생: {e}")

    # 다양한 셀렉터로 메뉴 수집 시도
    menu_result = try_menu_selectors(driver, wait)
    menus = []
    
    if menu_result:
        menu_elements, name_selector, price_selector = menu_result        
        for el in menu_elements:
            try:
                name = el.find_element(By.CSS_SELECTOR, name_selector).text
                price = el.find_element(By.CSS_SELECTOR, price_selector).text
                
                if validate_menu_item(name, price):
                    menus.append({"name": name.strip(), "price": price.strip()})
            except NoSuchElementException:
                continue
    else:
        # 일반 셀렉터가 실패하면 가격 기반 추출 시도
        logger.info("일반 셀렉터 실패, 가격 기반 추출 시도")
        menus = try_price_based_extraction(driver)
    
    if not menus:
        logger.warning("모든 방법으로 시도했지만 메뉴를 찾을 수 없습니다.")
        return None
    
    # 수집된 메뉴에서 중복 제거
    unique_menus = []
    seen_menus = set()
    for menu in menus:
        # 메뉴 이름을 고유한 키로 사용
        if menu['name'] not in seen_menus:
            unique_menus.append(menu)
            seen_menus.add(menu['name'])

    logger.info(f"총 {len(unique_menus)}개의 유효한 메뉴를 수집했습니다.")
    return unique_menus

def validate_review_text(text: str) -> bool:
    """리뷰 텍스트의 유효성을 검증합니다."""
    return bool(text and text.strip() and len(text.strip()) > 5)

def get_info_description(driver: webdriver.Chrome, wait: WebDriverWait) -> Optional[str]:
    """정보 탭의 업체 소개를 수집합니다."""
    # 다양한 정보 탭 이름 시도
    info_tab_names = ["정보", "Info", "상세정보"]
    tab_found = False
    
    for tab_name in info_tab_names:
        if click_tab(driver, wait, tab_name):
            tab_found = True
            logger.info(f"'{tab_name}' 탭을 찾았습니다.")
            break
    
    if not tab_found:
        logger.warning("정보 탭을 찾을 수 없습니다.")
        return None

    logger.info("업체 소개 정보 수집 시작")

    # 다양한 description 셀렉터 시도
    for selector in config.info_description_selectors:
        try:
            logger.debug(f"'{selector}' 셀렉터로 업체 소개 찾는 중...")
            description_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
            
            for element in description_elements:
                description_text = element.text.strip()
                return description_text
                    
        except TimeoutException:
            logger.debug(f"'{selector}' 셀렉터로 업체 소개를 찾지 못했습니다.")
            continue
        except Exception as e:
            logger.debug(f"'{selector}' 셀렉터 처리 중 오류: {e}")
            continue
    
    logger.warning("모든 셀렉터로 시도했지만 유효한 업체 소개를 찾지 못했습니다.")
    return None


def get_review_data(driver: webdriver.Chrome, wait: WebDriverWait) -> List[str]:
    """리뷰 탭의 모든 텍스트 리뷰를 수집합니다."""
    # 다양한 리뷰 탭 이름 시도
    review_tab_names = ["리뷰", "후기", "Review"]
    tab_found = False
    
    for tab_name in review_tab_names:
        if click_tab(driver, wait, tab_name):
            tab_found = True
            logger.info(f"'{tab_name}' 탭을 찾았습니다.")
            break
    
    if not tab_found:
        logger.warning("리뷰 탭을 찾을 수 없습니다.")
        return []

    logger.info("리뷰 정보 수집 시작")

    # 더보기 버튼 클릭
    click_count = click_more_button_until_done(
        driver, config.review_more_button_selector, config.max_review_clicks
    )
    logger.info(f"리뷰 더보기를 {click_count}번 클릭했습니다.")

    reviews = []
    try:
        logger.info(f"'{config.review_container_selector}' 컨테이너 대기 중...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.review_container_selector)))
        
        review_list_items = driver.find_elements(By.CSS_SELECTOR, f"{config.review_container_selector} > li")
        logger.info(f"총 {len(review_list_items)}개의 리뷰 항목을 찾았습니다.")

        for item in review_list_items:
            try:
                review_anchor = item.find_element(By.CSS_SELECTOR, config.review_item_selector)
                review_text = review_anchor.text.strip()
                
                if validate_review_text(review_text):
                    reviews.append(review_text)

            except NoSuchElementException:
                continue
        
        logger.info(f"총 {len(reviews)}개의 유효한 리뷰를 수집했습니다.")

    except TimeoutException:
        logger.error(f"'{config.review_container_selector}' 컨테이너를 찾지 못했습니다.")
    except Exception as e:
        logger.error(f"리뷰 텍스트 추출 중 오류 발생: {e}")
        
    return reviews


def load_existing_crawled_data(output_file: str) -> tuple[set, dict]:
    """기존에 크롤링된 업체 ID와 검색 키워드 매핑을 로드합니다."""
    crawled_place_ids = set()
    search_keyword_to_place_id = {}
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f_existing:
            for line in f_existing:
                try:
                    data = json.loads(line)
                    if 'place_id' in data:
                        place_id = data['place_id']
                        crawled_place_ids.add(place_id)
                        
                        # 검색 키워드가 있으면 매핑에 추가
                        if 'search_keyword' in data:
                            search_keyword_to_place_id[data['search_keyword']] = place_id
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"기존에 크롤링된 업체 ID {len(crawled_place_ids)}개를 로드했습니다.")
    logger.info(f"검색 키워드 매핑 {len(search_keyword_to_place_id)}개를 로드했습니다.")
    return crawled_place_ids, search_keyword_to_place_id

def extract_place_id_from_url(url: str) -> Optional[str]:
    """
URL에서 업체 ID를 추출합니다."""
    match = re.search(r"/place/(\d+)", url)
    return match.group(1) if match else None

def process_restaurant(driver: webdriver.Chrome, wait: WebDriverWait, restaurant_info: Dict[str, Any], crawled_place_ids: set, search_keyword_to_place_id: dict) -> Optional[Dict[str, Any]]:
    """개별 레스토랑 정보를 처리합니다."""
    title = restaurant_info.get("title", "")
    title = title.replace("&amp;", " ")
    road_address = restaurant_info.get("roadAddress", "")

    if not title or not road_address:
        logger.warning("제목 또는 주소가 비어있어 건너뜁니다.")
        return None

    short_address = " ".join(road_address.split()[:3])
    search_keyword = f"{title} {short_address}"
    
    # 검색 키워드를 통해 이미 크롤링된 업체인지 먼저 확인
    if search_keyword in search_keyword_to_place_id:
        existing_place_id = search_keyword_to_place_id[search_keyword]
        logger.info(f"이미 크롤링된 업체입니다 (검색키워드: {search_keyword}, ID: {existing_place_id}). 웹 요청을 건너뜁니다.")
        return None
    
    logger.info("=" * 80)
    logger.info(f"{title} ({search_keyword}) 크롤링 시작")

    try:
        encoded_keyword = quote(search_keyword)
        driver.get(f"{config.base_url}{encoded_keyword}")

        if not switch_to_entry_iframe(driver, WebDriverWait(driver, 5)):
            logger.warning(f"Iframe을 찾지 못해 {title}을(를) 건너뜁니다.")
            logger.info("=" * 80)
            return None

        # 업체 ID 추출
        place_id = extract_place_id_from_url(driver.current_url)
        if not place_id:
            logger.warning("URL에서 업체 ID를 찾지 못했습니다.")
            logger.info("=" * 80)
            return None
        
        # 이중 체크: place_id로도 한번 더 확인 (혹시 다른 검색키워드로 같은 업체가 크롤링되었을 경우)
        if place_id in crawled_place_ids:
            logger.info(f"이미 크롤링된 업체입니다 (ID: {place_id}). 건너뜁니다.")
            # 검색 키워드 매핑에 추가하여 다음번에는 웹 요청을 건너뛸 수 있도록 함
            search_keyword_to_place_id[search_keyword] = place_id
            logger.info("=" * 80)
            return None

        # 탭 컨테이너 대기
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.tab_container_selector)))
        except TimeoutException:
            logger.warning("탭 컨테이너를 찾을 수 없어 이 식당을 건너뜁니다.")
            logger.info("=" * 80)
            return None

        menu_data = get_menu_data(driver, wait)
        review_data = get_review_data(driver, wait)
        description_data = get_info_description(driver, wait)

        crawled_data = {
            "place_id": place_id,
            "search_keyword": search_keyword,
            **restaurant_info,
            "menus": menu_data if menu_data else [],
            "reviews": review_data if review_data else [],
            "description": description_data if description_data else "",
        }

        # 수집된 데이터 로깅
        logger.info(f"수집 결과 - 메뉴: {len(menu_data) if menu_data else 0}개, "
                    f"리뷰: {len(review_data) if review_data else 0}개, "
                    f"업체소개: {'있음' if description_data else '없음'}")

        logger.info(f"{title} 크롤링 완료")
        logger.info("=" * 80)
        return crawled_data

    except Exception as e:
        logger.error(f"'{title}' 크롤링 중 에러 발생: {e}")
        logger.info("=" * 80)
        return None

def main():
    """메인 크롤링 로직을 실행합니다."""
    logger.info("크롤링 시작")
    
    driver = get_driver()
    wait = WebDriverWait(driver, config.default_wait_time)
    crawled_place_ids, search_keyword_to_place_id = load_existing_crawled_data(OUTPUT_FILE)

    processed_count = 0
    success_count = 0
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            
            for line in f_in:
                try:
                    restaurant_info = json.loads(line.strip())
                    processed_count += 1
                    
                    crawled_data = process_restaurant(driver, wait, restaurant_info, crawled_place_ids, search_keyword_to_place_id)
                    
                    if crawled_data:
                        f_out.write(json.dumps(crawled_data, ensure_ascii=False) + '\n')
                        f_out.flush()  # 즉시 파일에 쓰기
                        # 성공한 크롤링 데이터를 메모리에도 업데이트
                        place_id = crawled_data['place_id']
                        search_keyword = crawled_data['search_keyword']
                        crawled_place_ids.add(place_id)
                        search_keyword_to_place_id[search_keyword] = place_id
                        success_count += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")
                    continue
                except Exception as e:
                    logger.error(f"예상치 못한 오류 발생: {e}")
                    continue

    finally:
        driver.quit()
        logger.info(f"크롤링 완료: 총 {processed_count}개 처리, {success_count}개 성공")

if __name__ == "__main__":
    main()
