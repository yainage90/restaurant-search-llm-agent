
import json
import re
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
# typing 라이브러리는 사용하지 않음 (Python 3.13)
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
    default_wait_time: float = 1  # 1.5 -> 0.8초로 단축
    max_review_clicks: int = 10
    scroll_wait_time: float = 1  # 스크롤 대기 시간
    click_wait_time: float = 1   # 클릭 후 대기 시간
    max_workers: int = 2  # 병렬 처리 워커 수 (너무 많으면 IP 차단 위험)
    
    # CSS Selectors
    search_iframe_id: str = "searchIframe"
    entry_iframe_id: str = "entryIframe"
    first_result_selector: str = "li.UEzoS:first-child .place_bluelink"
    tab_container_selector: str = ".Jxtsc"
    menu_more_button_selector: str = ".fvwqf"
    # Multiple menu item selectors for different layouts
    menu_item_selectors: list[str] = None
    menu_name_selectors: list[str] = None
    menu_price_selectors: list[str] = None
    # Sub-tab selectors (for menu category tabs etc.)
    review_more_button_selector: str = "a.fvwqf"
    review_container_selector: str = "ul#_review_list"
    review_item_selector: str = "div.pui__vn15t2 > a"
    # Restaurant info selectors
    info_description_selectors: list[str] = None
    
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
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/crawled_restaurants")
FAILED_QUERIES_FILE = os.path.join(BASE_DIR, "../data/crawl_failed_queries.txt")
MAX_RECORDS_PER_FILE = 1000

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# 파일 쓰기를 위한 전역 락
file_write_lock = threading.Lock()

def get_driver() -> webdriver.Chrome:
    """WebDriver를 설정하고 반환합니다."""
    logger.info("WebDriver 초기화 중...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    
    # 성능 최적화 옵션 추가
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")  # 이미지 로딩 비활성화
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    # 페이지 로딩 전략 설정
    prefs = {
        "profile.managed_default_content_settings.images": 2,  # 이미지 차단
        "profile.managed_default_content_settings.stylesheets": 2,  # CSS 차단 (선택적)
    }
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    # 페이지 로드 타임아웃 설정
    driver.set_page_load_timeout(5)
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
                time.sleep(config.click_wait_time)  # 더 짧은 대기 시간
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

def try_menu_selectors(driver: webdriver.Chrome, wait: WebDriverWait) -> tuple | None:
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

def try_price_based_extraction(driver: webdriver.Chrome) -> list[dict[str, str]]:
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
            time.sleep(config.scroll_wait_time)  # 스크롤 전용 대기 시간

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

def get_menu_data(driver: webdriver.Chrome, wait: WebDriverWait) -> list[dict[str, str]] | None:
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
        time.sleep(config.click_wait_time)  # 더 짧은 대기 시간

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
            time.sleep(config.scroll_wait_time)  # 설정 가능한 대기 시간
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

def get_info_description(driver: webdriver.Chrome, wait: WebDriverWait) -> str | None:
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


def get_review_data(driver: webdriver.Chrome, wait: WebDriverWait) -> list[str]:
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


def load_existing_crawled_data(output_dir: str) -> tuple[set, dict]:
    """기존에 크롤링된 업체 ID와 검색 키워드 매핑을 로드합니다."""
    crawled_place_ids = set()
    search_keyword_to_place_id = {}
    
    if os.path.exists(output_dir):
        # 모든 part 파일을 읽어들임
        for filename in os.listdir(output_dir):
            if filename.startswith('part-') and filename.endswith('.jsonl'):
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f_existing:
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

def load_failed_keywords(failed_keywords_file: str) -> set[str]:
    """실패한 검색어 목록을 로드합니다."""
    failed_keywords = set()
    
    if os.path.exists(failed_keywords_file):
        with open(failed_keywords_file, 'r', encoding='utf-8') as f:
            for line in f:
                keyword = line.strip()
                if keyword:
                    failed_keywords.add(keyword)
    
    logger.info(f"실패한 검색어 {len(failed_keywords)}개를 로드했습니다.")
    return failed_keywords

def save_failed_keyword(failed_keywords_file: str, keyword: str):
    """실패한 검색어를 파일에 기록합니다. (스레드 안전)"""
    with file_write_lock:
        os.makedirs(os.path.dirname(failed_keywords_file), exist_ok=True)
        with open(failed_keywords_file, 'a', encoding='utf-8') as f:
            f.write(keyword + '\n')
        logger.info(f"실패한 검색어 기록: {keyword}")

def extract_place_id_from_url(url: str) -> str | None:
    """
URL에서 업체 ID를 추출합니다."""
    match = re.search(r"/place/(\d+)", url)
    return match.group(1) if match else None

def process_restaurant(driver: webdriver.Chrome, wait: WebDriverWait, restaurant_info: dict[str, any], crawled_place_ids: set, search_keyword_to_place_id: dict, failed_keywords: set) -> tuple[dict[str, any] | None, bool]:
    """개별 레스토랑 정보를 처리합니다."""
    title = restaurant_info.get("title", "")
    title = title.replace("&amp;", " ")
    road_address = restaurant_info.get("roadAddress", "")

    if not title or not road_address:
        logger.warning("제목 또는 주소가 비어있어 건너뜁니다.")
        return None, False

    short_address = " ".join(road_address.split()[:3])
    search_keyword = f"{title} {short_address}"
    
    # 실패한 검색어인지 먼저 확인
    if search_keyword in failed_keywords:
        logger.info(f"이미 실패한 검색어입니다: {search_keyword}. 건너뜁니다.")
        return None, False
    
    # 검색 키워드를 통해 이미 크롤링된 업체인지 확인
    if search_keyword in search_keyword_to_place_id:
        existing_place_id = search_keyword_to_place_id[search_keyword]
        logger.info(f"이미 크롤링된 업체입니다 (검색키워드: {search_keyword}, ID: {existing_place_id}). 웹 요청을 건너뜁니다.")
        return None, False
    
    logger.info("=" * 80)
    logger.info(f"{title} ({search_keyword}) 크롤링 시작")

    try:
        encoded_keyword = quote(search_keyword)
        driver.get(f"{config.base_url}{encoded_keyword}")

        if not switch_to_entry_iframe(driver, WebDriverWait(driver, 5)):
            logger.warning(f"Iframe을 찾지 못해 {title}을(를) 건너뜁니다.")
            logger.info("=" * 80)
            return None, True  # 실패 상황으로 기록

        # 업체 ID 추출
        place_id = extract_place_id_from_url(driver.current_url)
        if not place_id:
            logger.warning("URL에서 업체 ID를 찾지 못했습니다.")
            logger.info("=" * 80)
            return None, True  # 실패 상황으로 기록
        
        # 이중 체크: place_id로도 한번 더 확인 (혹시 다른 검색키워드로 같은 업체가 크롤링되었을 경우)
        if place_id in crawled_place_ids:
            logger.info(f"이미 크롤링된 업체입니다 (ID: {place_id}). 건너뜁니다.")
            # 검색 키워드 매핑에 추가하여 다음번에는 웹 요청을 건너뛸 수 있도록 함
            search_keyword_to_place_id[search_keyword] = place_id
            logger.info("=" * 80)
            return None, False  # 성공적으로 건너뛀

        # 탭 컨테이너 대기
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.tab_container_selector)))
        except TimeoutException:
            logger.warning("탭 컨테이너를 찾을 수 없어 이 식당을 건너뜁니다.")
            logger.info("=" * 80)
            return None, False # 실패 상황으로 기록

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
        return crawled_data, False  # 성공

    except Exception as e:
        logger.error(f"'{title}' 크롤링 중 에러 발생: {e}")
        logger.info("=" * 80)
        return None, True  # 실패 상황으로 기록

def get_current_part_file_path(output_dir: str) -> str:
    """현재 사용할 part 파일 경로를 반환합니다. (스레드 안전)"""
    with file_write_lock:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 기존 part 파일들 중 가장 높은 번호 찾기
        max_part_num = -1
        current_file_record_count = 0
        
        for filename in os.listdir(output_dir):
            if filename.startswith('part-') and filename.endswith('.jsonl'):
                try:
                    part_num = int(filename.split('-')[1].split('.')[0])
                    max_part_num = max(max_part_num, part_num)
                except (ValueError, IndexError):
                    continue
        
        # 현재 파일의 레코드 수 확인
        if max_part_num >= 0:
            current_file_path = os.path.join(output_dir, f"part-{max_part_num:05d}.jsonl")
            if os.path.exists(current_file_path):
                with open(current_file_path, 'r', encoding='utf-8') as f:
                    current_file_record_count = sum(1 for line in f if line.strip())
        
        # 새 파일이 필요한지 확인
        if max_part_num < 0 or current_file_record_count >= MAX_RECORDS_PER_FILE:
            max_part_num += 1
        
        return os.path.join(output_dir, f"part-{max_part_num:05d}.jsonl")

def process_batch_worker(batch_data: list[dict], worker_id: int, shared_state: dict) -> dict:
    """개별 워커에서 배치 처리를 수행합니다."""
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.info(f"워커 {worker_id} 시작: {len(batch_data)}개 아이템 처리")
    
    driver = get_driver()
    wait = WebDriverWait(driver, config.default_wait_time)
    
    processed_count = 0
    success_count = 0
    failed_keywords = []
    results = []
    
    try:
        for restaurant_info in batch_data:
            crawled_data, should_record_failure = process_restaurant(
                driver, wait, restaurant_info, 
                shared_state['crawled_place_ids'], 
                shared_state['search_keyword_to_place_id'], 
                shared_state['failed_keywords']
            )
            
            processed_count += 1
            
            # 실패한 경우 기록
            if should_record_failure and crawled_data is None:
                title = restaurant_info.get("title", "")
                title = title.replace("&amp;", " ")
                road_address = restaurant_info.get("roadAddress", "")
                if title and road_address:
                    short_address = " ".join(road_address.split()[:3])
                    search_keyword = f"{title} {short_address}"
                    failed_keywords.append(search_keyword)
            
            if crawled_data:
                results.append(crawled_data)
                success_count += 1
    
    except Exception as e:
        worker_logger.error(f"워커 {worker_id} 오류: {e}")
    finally:
        driver.quit()
    
    worker_logger.info(f"워커 {worker_id} 완료: {processed_count}개 처리, {success_count}개 성공")
    return {
        'worker_id': worker_id,
        'processed_count': processed_count,
        'success_count': success_count,
        'results': results,
        'failed_keywords': failed_keywords
    }

def save_batch_results(results: list[dict], output_dir: str) -> int:
    """배치 결과를 파일에 저장합니다. (스레드 안전)"""
    if not results:
        return 0
    
    saved_count = 0
    
    # 전체 저장 과정을 락으로 보호
    with file_write_lock:
        try:
            # 모든 결과를 한 번에 저장하여 파일 핸들 경쟁 상태 방지
            current_file_record_count = 0
            current_output_file = None
            current_file_handle = None
            
            for result in results:
                # 새 파일이 필요한지 확인
                if current_file_handle is None or current_file_record_count >= MAX_RECORDS_PER_FILE:
                    if current_file_handle:
                        current_file_handle.close()
                    
                    current_output_file = get_current_part_file_path(output_dir)
                    current_file_handle = open(current_output_file, 'a', encoding='utf-8')
                    
                    # 기존 파일의 레코드 수 다시 확인 (락 안에서)
                    if os.path.exists(current_output_file):
                        with open(current_output_file, 'r', encoding='utf-8') as temp_f:
                            current_file_record_count = sum(1 for temp_line in temp_f if temp_line.strip())
                    else:
                        current_file_record_count = 0
                    
                    logger.info(f"새 출력 파일 사용: {current_output_file} (현재 레코드 수: {current_file_record_count})")
                
                # 데이터 쓰기
                current_file_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                current_file_handle.flush()  # 즉시 파일에 쓰기
                current_file_record_count += 1
                saved_count += 1
                        
            if current_file_handle:
                current_file_handle.close()
                
        except Exception as e:
            logger.error(f"배치 결과 저장 중 오류: {e}")
            if 'current_file_handle' in locals() and current_file_handle:
                current_file_handle.close()
    
    return saved_count

def main():
    """메인 크롤링 로직을 실행합니다 (병렬 처리 버전)."""
    logger.info("크롤링 시작 (병렬 처리 모드)")
    
    crawled_place_ids, search_keyword_to_place_id = load_existing_crawled_data(OUTPUT_DIR)
    failed_keywords = load_failed_keywords(FAILED_QUERIES_FILE)
    
    # 공유 상태
    shared_state = {
        'crawled_place_ids': crawled_place_ids,
        'search_keyword_to_place_id': search_keyword_to_place_id,
        'failed_keywords': failed_keywords
    }
    
    try:
        # INPUT_FILE을 읽고 랜덤으로 셔플
        logger.info(f"INPUT_FILE 읽는 중: {INPUT_FILE}")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        logger.info(f"총 {len(lines)}개 레스토랑 데이터 로드됨")
        random.shuffle(lines)
        logger.info("레스토랑 데이터를 랜덤으로 셔플했습니다")
        
        # 레스토랑 데이터를 배치로 분할
        restaurant_data = []
        for line in lines:
            try:
                restaurant_info = json.loads(line.strip())
                restaurant_data.append(restaurant_info)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {e}")
                continue
        
        # 워커 수에 맞게 배치 분할
        batch_size = max(1, len(restaurant_data) // (config.max_workers * 2))  # 워커당 2배치 정도
        batches = [restaurant_data[i:i + batch_size] for i in range(0, len(restaurant_data), batch_size)]
        
        logger.info(f"{len(batches)}개 배치로 분할됨 (배치당 평균 {batch_size}개 아이템)")
        logger.info(f"{config.max_workers}개 워커로 병렬 처리 시작")
        
        total_processed = 0
        total_success = 0
        all_failed_keywords = set(failed_keywords)
        
        # ThreadPoolExecutor 사용 (프로세스 풀보다 메모리 효율적)
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # 배치를 워커에 분배
            futures = []
            for i, batch in enumerate(batches):
                future = executor.submit(process_batch_worker, batch, i, shared_state)
                futures.append(future)
            
            # 결과 수집 및 처리
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5분 타임아웃
                    total_processed += result['processed_count']
                    total_success += result['success_count']
                    
                    # 실패한 키워드 업데이트
                    for keyword in result['failed_keywords']:
                        if keyword not in all_failed_keywords:
                            save_failed_keyword(FAILED_QUERIES_FILE, keyword)
                            all_failed_keywords.add(keyword)
                    
                    # 결과 저장
                    if result['results']:
                        saved_count = save_batch_results(result['results'], OUTPUT_DIR)
                        logger.info(f"워커 {result['worker_id']} 결과 저장 완료: {saved_count}개")
                        
                        # 성공한 크롤링 데이터를 공유 상태에 업데이트
                        for crawled_data in result['results']:
                            place_id = crawled_data['place_id']
                            search_keyword = crawled_data['search_keyword']
                            shared_state['crawled_place_ids'].add(place_id)
                            shared_state['search_keyword_to_place_id'][search_keyword] = place_id
                    
                except Exception as e:
                    logger.error(f"워커 결과 처리 중 오류: {e}")
        
        logger.info(f"병렬 크롤링 완료: 총 {total_processed}개 처리, {total_success}개 성공")
        
    except Exception as e:
        logger.error(f"메인 프로세스 오류: {e}")

def main_sequential():
    """기존 순차 처리 방식 (백업용)."""
    logger.info("크롤링 시작 (순차 처리 모드)")
    
    driver = get_driver()
    wait = WebDriverWait(driver, config.default_wait_time)
    crawled_place_ids, search_keyword_to_place_id = load_existing_crawled_data(OUTPUT_DIR)
    failed_keywords = load_failed_keywords(FAILED_QUERIES_FILE)

    processed_count = 0
    success_count = 0
    current_file_record_count = 0
    current_output_file = None
    current_file_handle = None
    
    try:
        # INPUT_FILE을 읽고 랜덤으로 셔플
        logger.info(f"INPUT_FILE 읽는 중: {INPUT_FILE}")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        logger.info(f"총 {len(lines)}개 레스토랑 데이터 로드됨")
        random.shuffle(lines)
        logger.info("레스토랑 데이터를 랜덤으로 셔플했습니다")
        
        for line in lines:
            try:
                restaurant_info = json.loads(line.strip())
                processed_count += 1
                
                crawled_data, should_record_failure = process_restaurant(driver, wait, restaurant_info, crawled_place_ids, search_keyword_to_place_id, failed_keywords)
                
                # 실패한 경우 기록
                if should_record_failure and crawled_data is None:
                    title = restaurant_info.get("title", "")
                    title = title.replace("&amp;", " ")
                    road_address = restaurant_info.get("roadAddress", "")
                    if title and road_address:
                        short_address = " ".join(road_address.split()[:3])
                        search_keyword = f"{title} {short_address}"
                        save_failed_keyword(FAILED_QUERIES_FILE, search_keyword)
                        failed_keywords.add(search_keyword)
                
                if crawled_data:
                    # 새 파일이 필요한지 확인
                    if current_file_handle is None or current_file_record_count >= MAX_RECORDS_PER_FILE:
                        if current_file_handle:
                            current_file_handle.close()
                        
                        current_output_file = get_current_part_file_path(OUTPUT_DIR)
                        current_file_handle = open(current_output_file, 'a', encoding='utf-8')
                        
                        # 기존 파일의 레코드 수 확인
                        if os.path.exists(current_output_file):
                            with open(current_output_file, 'r', encoding='utf-8') as temp_f:
                                current_file_record_count = sum(1 for temp_line in temp_f if temp_line.strip())
                        else:
                            current_file_record_count = 0
                        
                        logger.info(f"새 출력 파일 사용: {current_output_file} (현재 레코드 수: {current_file_record_count})")
                    
                    # 데이터 쓰기
                    current_file_handle.write(json.dumps(crawled_data, ensure_ascii=False) + '\n')
                    current_file_handle.flush()  # 즉시 파일에 쓰기
                    current_file_record_count += 1
                    
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
        if current_file_handle:
            current_file_handle.close()
        driver.quit()
        logger.info(f"크롤링 완료: 총 {processed_count}개 처리, {success_count}개 성공")

if __name__ == "__main__":
    # 환경변수로 실행 모드 선택 가능
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--sequential":
        main_sequential()
    else:
        main()  # 기본값: 병렬 처리
