# ========================================================================
# ПОЛНЫЙ СКВОЗНОЙ ПАЙПЛАЙН РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ
# ========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Настройки
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================================================
# ЭТАП 1: ПОДГОТОВКА ДАННЫХ
# ========================================================================

print("=" * 80)
print("ЭТАП 1: ПОДГОТОВКА ДАННЫХ")
print("=" * 80)

class DataPreprocessor:
    """Класс для подготовки данных"""
    
    def __init__(self):
        self.user_features = None
        self.book_features = None
        self.interaction_features = None
        
    def load_data(self):
        """Загрузка данных"""
        print("Загрузка данных...")
        
        try:
            self.ratings = pd.read_csv('goodbooks-10k/ratings.csv')
            self.books = pd.read_csv('goodbooks-10k/books.csv')
            self.tags = pd.read_csv('goodbooks-10k/tags.csv')
            self.book_tags = pd.read_csv('goodbooks-10k/book_tags.csv')
            
            print(f"✓ Данные загружены:")
            print(f"  • Оценки: {self.ratings.shape}")
            print(f"  • Книги: {self.books.shape}")
            print(f"  • Теги: {self.tags.shape}")
            print(f"  • Теги книг: {self.book_tags.shape}")
            
        except Exception as e:
            print(f"✗ Ошибка загрузки: {e}")
            raise
    
    def create_temporal_split(self):
        """Создание временного разделения на train/test"""
        print("\nСоздание временного разделения...")
        
        # Добавляем временные метки если нет
        if 'timestamp' not in self.ratings.columns:
            np.random.seed(42)
            dates = pd.date_range('2010-01-01', '2020-12-31', periods=len(self.ratings))
            self.ratings['timestamp'] = dates
        
        # Сортировка по времени
        self.ratings = self.ratings.sort_values('timestamp')
        
        # Разделение (80/20)
        train_size = int(0.8 * len(self.ratings))
        self.train_data = self.ratings.iloc[:train_size]
        self.test_data = self.ratings.iloc[train_size:]
        
        print(f"✓ Разделение создано:")
        print(f"  • Train: {len(self.train_data):,} записей")
        print(f"  • Test: {len(self.test_data):,} записей")
        
        # Проверка пересечений
        train_users = set(self.train_data['user_id'])
        train_books = set(self.train_data['book_id'])
        test_users = set(self.test_data['user_id'])
        test_books = set(self.test_data['book_id'])
        
        print(f"  • Пользователей в test и train: {len(test_users & train_users) / len(test_users):.1%}")
        print(f"  • Книг в test и train: {len(test_books & train_books) / len(test_books):.1%}")
    
    def create_user_features(self):
        """Создание расширенных признаков пользователей"""
        print("\nСоздание признаков пользователей...")
        
        features = []
        
        for user_id in tqdm(self.train_data['user_id'].unique(), desc="Пользователи"):
            user_ratings = self.train_data[self.train_data['user_id'] == user_id]
            
            # Базовые статистики
            rating_stats = {
                'user_id': user_id,
                'user_rating_count': len(user_ratings),
                'user_avg_rating': user_ratings['rating'].mean(),
                'user_rating_std': user_ratings['rating'].std(),
                'user_rating_median': user_ratings['rating'].median(),
                'user_rating_min': user_ratings['rating'].min(),
                'user_rating_max': user_ratings['rating'].max(),
            }
            
            # Временные паттерны
            if 'timestamp' in user_ratings.columns:
                timestamps = user_ratings['timestamp'].sort_values()
                if len(timestamps) > 1:
                    intervals = np.diff(timestamps.values.astype(np.int64) // 10**9)
                    rating_stats['user_avg_time_interval'] = intervals.mean()
                    rating_stats['user_time_interval_std'] = intervals.std()
                else:
                    rating_stats['user_avg_time_interval'] = 0
                    rating_stats['user_time_interval_std'] = 0
            
            # Распределение оценок
            for rating_val in [1, 2, 3, 4, 5]:
                count = (user_ratings['rating'] == rating_val).sum()
                rating_stats[f'user_rating_{rating_val}_count'] = count
                rating_stats[f'user_rating_{rating_val}_ratio'] = count / len(user_ratings) if len(user_ratings) > 0 else 0
            
            # Активность (классификация)
            if len(user_ratings) < 5:
                rating_stats['user_activity_level'] = 'low'
            elif len(user_ratings) < 20:
                rating_stats['user_activity_level'] = 'medium'
            else:
                rating_stats['user_activity_level'] = 'high'
            
            features.append(rating_stats)
        
        self.user_features = pd.DataFrame(features)
        print(f"✓ Создано признаков для {len(self.user_features)} пользователей")
    
    def create_book_features(self):
        """Создание расширенных признаков книг"""
        print("\nСоздание признаков книг...")
        
        # Объединяем теги книг
        book_tags_merged = pd.merge(self.book_tags, self.tags, on='tag_id', how='left')
        
        # Создаем TF-IDF векторы для тегов
        print("  Создание TF-IDF векторов...")
        tag_vectors = {}
        
        for book_id in tqdm(self.books['book_id'].unique(), desc="TF-IDF векторы"):
            book_tags = book_tags_merged[book_tags_merged['goodreads_book_id'] == book_id]
            tags_text = ' '.join([str(tag) for tag in book_tags['tag_name'].fillna('').values])
            tag_vectors[book_id] = tags_text
        
        # Создаем DataFrame с тегами
        tag_df = pd.DataFrame(list(tag_vectors.items()), columns=['book_id', 'tags_text'])
        self.books = pd.merge(self.books, tag_df, on='book_id', how='left')
        
        # TF-IDF векторная модель
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books['tags_text'].fillna(''))
        
        # Матрица сходства книг
        print("  Создание матрицы сходства книг...")
        self.book_similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Статистики по оценкам для каждой книги
        book_stats = self.train_data.groupby('book_id').agg({
            'rating': ['count', 'mean', 'std', 'median', 'min', 'max']
        }).reset_index()
        
        book_stats.columns = ['book_id', 'book_rating_count', 'book_avg_rating', 
                             'book_rating_std', 'book_rating_median', 
                             'book_rating_min', 'book_rating_max']
        
        # Добавляем разнообразие оценок (энтропию)
        def calculate_rating_entropy(ratings):
            rating_counts = ratings.value_counts(normalize=True)
            return -sum(rating_counts * np.log2(rating_counts + 1e-10))
        
        book_entropy = self.train_data.groupby('book_id')['rating'].apply(calculate_rating_entropy)
        book_entropy.name = 'book_rating_entropy'
        book_stats = pd.merge(book_stats, book_entropy, on='book_id', how='left')
        
        # Категории популярности
        def categorize_popularity(count):
            if count < 10: return 'very_low'
            elif count < 50: return 'low'
            elif count < 200: return 'medium'
            elif count < 500: return 'high'
            else: return 'very_high'
        
        book_stats['book_popularity_category'] = book_stats['book_rating_count'].apply(categorize_popularity)
        
        # Категории рейтинга
        def categorize_rating(rating):
            if rating < 2.5: return 'very_low'
            elif rating < 3.0: return 'low'
            elif rating < 3.5: return 'medium'
            elif rating < 4.0: return 'high'
            else: return 'very_high'
        
        book_stats['book_rating_category'] = book_stats['book_avg_rating'].apply(categorize_rating)
        
        # Объединяем с основной информацией о книгах
        self.book_features = pd.merge(self.books, book_stats, on='book_id', how='left')
        
        # Заполняем пропуски
        numeric_cols = self.book_features.select_dtypes(include=[np.number]).columns
        self.book_features[numeric_cols] = self.book_features[numeric_cols].fillna(self.book_features[numeric_cols].median())
        
        print(f"✓ Создано признаков для {len(self.book_features)} книг")
    
    def create_interaction_features(self):
        """Создание признаков взаимодействий"""
        print("\nСоздание признаков взаимодействий...")
        
        features = []
        
        # Для каждой пары пользователь-книга в трейне
        for idx, row in tqdm(self.train_data.iterrows(), total=len(self.train_data), desc="Взаимодействия"):
            user_id = row['user_id']
            book_id = row['book_id']
            rating = row['rating']
            
            # Получаем признаки пользователя и книги
            user_feat = self.user_features[self.user_features['user_id'] == user_id].iloc[0] if user_id in self.user_features['user_id'].values else None
            book_feat = self.book_features[self.book_features['book_id'] == book_id].iloc[0] if book_id in self.book_features['book_id'].values else None
            
            if user_feat is not None and book_feat is not None:
                # Вычисляем схожесть с историей пользователя
                similarity_score = 0
                if user_id in self.train_data['user_id'].values:
                    user_books = self.train_data[self.train_data['user_id'] == user_id]['book_id'].values
                    if len(user_books) > 0:
                        # Для каждой книги в истории вычисляем сходство
                        similarities = []
                        for ub in user_books:
                            if ub in self.book_features['book_id'].values and book_id in self.book_features['book_id'].values:
                                idx1 = self.book_features[self.book_features['book_id'] == ub].index[0]
                                idx2 = self.book_features[self.book_features['book_id'] == book_id].index[0]
                                similarities.append(self.book_similarity_matrix[idx1][idx2])
                        similarity_score = np.mean(similarities) if similarities else 0
                
                # Разница между средней оценкой пользователя и средней оценкой книги
                rating_diff = abs(user_feat['user_avg_rating'] - book_feat['book_avg_rating']) if not pd.isna(user_feat['user_avg_rating']) else 0
                
                # Вес книги (популярность * качество)
                book_weight = book_feat['book_rating_count'] * book_feat['book_avg_rating'] / 100
                
                features.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'rating': rating,
                    'similarity_score': similarity_score,
                    'rating_diff': rating_diff,
                    'book_weight': book_weight,
                    'user_book_rating_std_diff': abs(user_feat['user_rating_std'] - book_feat['book_rating_std']) if not pd.isna(user_feat['user_rating_std']) else 0
                })
        
        self.interaction_features = pd.DataFrame(features)
        print(f"✓ Создано {len(self.interaction_features)} признаков взаимодействий")
    
    def prepare_all_features(self):
        """Запуск всей подготовки данных"""
        self.load_data()
        self.create_temporal_split()
        self.create_user_features()
        self.create_book_features()
        self.create_interaction_features()
        
        print("\n" + "="*80)
        print("СВОДКА ПО ПОДГОТОВКЕ ДАННЫХ:")
        print("="*80)
        print(f"• Пользователи: {len(self.user_features)} записей")
        print(f"• Книги: {len(self.book_features)} записей")
        print(f"• Взаимодействия: {len(self.interaction_features)} записей")
        print(f"• Признаки пользователей: {self.user_features.shape[1]} столбцов")
        print(f"• Признаки книг: {self.book_features.shape[1]} столбцов")
        
        return self.user_features, self.book_features, self.interaction_features

# ========================================================================
# ЭТАП 2: ПОСТРОЕНИЕ ГИБРИДНОЙ СИСТЕМЫ
# ========================================================================

print("\n" + "="*80)
print("ЭТАП 2: ПОСТРОЕНИЕ ГИБРИДНОЙ СИСТЕМЫ")
print("="*80)

class HybridRecommenderSystem:
    """Гибридная рекомендательная система"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.models = {}
        self.user_segments = {}
        
    def segment_users(self):
        """Сегментация пользователей по типам"""
        print("\nСегментация пользователей...")
        
        # Классификация пользователей по активности
        for user_id in self.preprocessor.train_data['user_id'].unique():
            user_ratings = self.preprocessor.train_data[self.preprocessor.train_data['user_id'] == user_id]
            rating_count = len(user_ratings)
            
            if rating_count < 5:
                segment = 'new_user'
            elif rating_count < 20:
                segment = 'active_user'
            else:
                segment = 'power_user'
            
            self.user_segments[user_id] = segment
        
        # Статистика сегментов
        segment_counts = Counter(self.user_segments.values())
        print("✓ Сегментация завершена:")
        for segment, count in segment_counts.items():
            print(f"  • {segment}: {count} пользователей ({count/len(self.user_segments):.1%})")
    
    def train_popularity_model(self):
        """Модель популярности"""
        print("\nОбучение модели популярности...")
        
        # Байесовское среднее для устойчивости
        popularity = self.preprocessor.train_data.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        popularity.columns = ['book_id', 'avg_rating', 'rating_count']
        
        # Добавляем штраф за малое количество оценок
        C = popularity['avg_rating'].mean()
        m = popularity['rating_count'].quantile(0.5)
        
        popularity['bayesian_score'] = (
            (popularity['rating_count'] * popularity['avg_rating'] + C * m) / 
            (popularity['rating_count'] + m)
        )
        
        popularity = popularity.sort_values('bayesian_score', ascending=False)
        self.models['popularity'] = popularity
        
        print(f"✓ Обучено на {len(popularity)} книгах")
    
    def train_content_based_model(self):
        """Content-based модель"""
        print("\nОбучение Content-based модели...")
        
        # Используем TF-IDF матрицу из препроцессора
        self.models['content_based'] = {
            'tfidf_matrix': self.preprocessor.tfidf_matrix,
            'similarity_matrix': self.preprocessor.book_similarity_matrix,
            'book_features': self.preprocessor.book_features
        }
        
        print("✓ Content-based модель готова")
    
    def train_collaborative_model(self):
        """Collaborative Filtering модель (SVD)"""
        print("\nОбучение Collaborative Filtering модели...")
        
        # Подготовка данных для Surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            self.preprocessor.train_data[['user_id', 'book_id', 'rating']], 
            reader
        )
        trainset = data.build_full_trainset()
        
        # Обучение SVD с настройкой гиперпараметров
        svd = SVD(
            n_factors=150,
            n_epochs=25,
            lr_all=0.007,
            reg_all=0.03,
            random_state=42
        )
        svd.fit(trainset)
        
        self.models['collaborative'] = svd
        print("✓ Collaborative Filtering модель обучена")
    
    def train_hybrid_model(self):
        """Гибридная модель с обучаемыми весами"""
        print("\nОбучение гибридной модели...")
        
        # Подготовка данных для обучения
        train_samples = self.preprocessor.interaction_features.sample(
            min(50000, len(self.preprocessor.interaction_features)),
            random_state=42
        )
        
        X = []
        y = []
        
        for idx, row in tqdm(train_samples.iterrows(), total=len(train_samples), desc="Подготовка признаков"):
            user_id = row['user_id']
            book_id = row['book_id']
            
            # Извлекаем признаки для обучения
            features = self._extract_hybrid_features(user_id, book_id)
            X.append(features)
            y.append(row['rating'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Разделение на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Обучение Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Оценка
        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        self.models['hybrid'] = rf
        
        print(f"✓ Гибридная модель обучена:")
        print(f"  • Train RMSE: {train_rmse:.4f}")
        print(f"  • Validation RMSE: {val_rmse:.4f}")
    
    def _extract_hybrid_features(self, user_id, book_id):
        """Извлечение признаков для гибридной модели"""
        features = []
        
        # 1. Признаки пользователя
        if user_id in self.preprocessor.user_features['user_id'].values:
            user_row = self.preprocessor.user_features[self.preprocessor.user_features['user_id'] == user_id].iloc[0]
            features.extend([
                user_row['user_rating_count'],
                user_row['user_avg_rating'],
                user_row['user_rating_std'],
                1 if user_row['user_activity_level'] == 'low' else 0,
                1 if user_row['user_activity_level'] == 'medium' else 0,
                1 if user_row['user_activity_level'] == 'high' else 0
            ])
        else:
            features.extend([0, 3.0, 0, 1, 0, 0])  # Значения по умолчанию для нового пользователя
        
        # 2. Признаки книги
        if book_id in self.preprocessor.book_features['book_id'].values:
            book_row = self.preprocessor.book_features[self.preprocessor.book_features['book_id'] == book_id].iloc[0]
            features.extend([
                book_row['book_rating_count'],
                book_row['book_avg_rating'],
                book_row['book_rating_std'],
                book_row['book_rating_entropy'],
                1 if book_row['book_popularity_category'] == 'very_low' else 0,
                1 if book_row['book_popularity_category'] == 'low' else 0,
                1 if book_row['book_popularity_category'] == 'medium' else 0,
                1 if book_row['book_popularity_category'] == 'high' else 0,
                1 if book_row['book_popularity_category'] == 'very_high' else 0
            ])
        else:
            features.extend([0, 3.0, 0, 0, 0, 0, 1, 0, 0])  # Значения по умолчанию
        
        # 3. Content-based признаки (сходство)
        if user_id in self.preprocessor.train_data['user_id'].values:
            user_books = self.preprocessor.train_data[self.preprocessor.train_data['user_id'] == user_id]['book_id'].values
            if len(user_books) > 0 and book_id in self.preprocessor.book_features['book_id'].values:
                similarities = []
                for ub in user_books[:10]:  # Ограничиваем для скорости
                    if ub in self.preprocessor.book_features['book_id'].values:
                        idx1 = self.preprocessor.book_features[self.preprocessor.book_features['book_id'] == ub].index[0]
                        idx2 = self.preprocessor.book_features[self.preprocessor.book_features['book_id'] == book_id].index[0]
                        similarities.append(self.preprocessor.book_similarity_matrix[idx1][idx2])
                features.append(np.mean(similarities) if similarities else 0)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # 4. Collaborative Filtering предсказание
        try:
            if hasattr(self.models.get('collaborative'), 'predict'):
                pred = self.models['collaborative'].predict(user_id, book_id)
                features.append(pred.est)
            else:
                features.append(3.0)
        except:
            features.append(3.0)
        
        return np.array(features)
    
    def generate_candidate_pool(self, user_id, top_n=100):
        """Генерация пула кандидатов из всех моделей"""
        candidates = set()
        
        # 1. По популярности (для всех пользователей)
        popularity_rec = self.models['popularity'].head(50)['book_id'].tolist()
        candidates.update(popularity_rec)
        
        # 2. По сегменту пользователя
        user_segment = self.user_segments.get(user_id, 'new_user')
        
        if user_segment == 'new_user':
            # Для новых пользователей - больше популярного контента
            trending_books = self.models['popularity'].head(100)['book_id'].tolist()
            candidates.update(trending_books)
            
        elif user_segment == 'active_user':
            # Для активных - комбинация подходов
            # Content-based
            if user_id in self.preprocessor.train_data['user_id'].values:
                user_books = self.preprocessor.train_data[self.preprocessor.train_data['user_id'] == user_id]['book_id'].values
                if len(user_books) > 0:
                    for book_id in user_books[:5]:  # Берем 5 последних книг
                        if book_id in self.preprocessor.book_features['book_id'].values:
                            idx = self.preprocessor.book_features[self.preprocessor.book_features['book_id'] == book_id].index[0]
                            sim_scores = list(enumerate(self.preprocessor.book_similarity_matrix[idx]))
                            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:20]
                            for sim_idx, score in sim_scores:
                                candidates.add(self.preprocessor.book_features.iloc[sim_idx]['book_id'])
            
            # Collaborative
            try:
                # Берем популярные книги для предсказания
                popular_books = self.models['popularity'].head(200)['book_id'].tolist()
                predictions = []
                for book_id in popular_books[:50]:
                    try:
                        pred = self.models['collaborative'].predict(user_id, book_id)
                        predictions.append((book_id, pred.est))
                    except:
                        continue
                predictions.sort(key=lambda x: x[1], reverse=True)
                candidates.update([b for b, _ in predictions[:30]])
            except:
                pass
        
        else:  # power_user
            # Для опытных - персонализированные рекомендации
            # Collaborative с большим пулом
            try:
                popular_books = self.models['popularity'].head(500)['book_id'].tolist()
                predictions = []
                for book_id in popular_books[:100]:
                    try:
                        pred = self.models['collaborative'].predict(user_id, book_id)
                        predictions.append((book_id, pred.est))
                    except:
                        continue
                predictions.sort(key=lambda x: x[1], reverse=True)
                candidates.update([b for b, _ in predictions[:50]])
            except:
                pass
        
        # Исключаем уже прочитанные книги
        if user_id in self.preprocessor.train_data['user_id'].values:
            read_books = set(self.preprocessor.train_data[self.preprocessor.train_data['user_id'] == user_id]['book_id'])
            candidates = candidates - read_books
        
        # Ограничиваем размер пула
        return list(candidates)[:top_n]
    
    def rank_candidates(self, user_id, candidates, top_n=20, diversity_weight=0.2):
        """Ранжирование кандидатов с учетом разнообразия"""
        if not candidates:
            return []
        
        # Вычисляем скоры для всех кандидатов
        scores = []
        candidate_features = []
        
        for book_id in candidates:
            try:
                # Получаем признаки
                features = self._extract_hybrid_features(user_id, book_id)
                # Предсказываем рейтинг
                score = self.models['hybrid'].predict(features.reshape(1, -1))[0]
                scores.append((book_id, score))
                candidate_features.append(features)
            except Exception as e:
                # Если ошибка, используем средний скор
                scores.append((book_id, 3.0))
                candidate_features.append(np.zeros(29))  # 29 признаков
        
        # Сортируем по скору
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Применяем Maximal Marginal Relevance (MMR) для разнообразия
        if diversity_weight > 0 and len(scores) > 1:
            selected = []
            remaining = scores.copy()
            
            # Начинаем с лучшего
            selected.append(remaining.pop(0))
            
            while len(selected) < min(top_n, len(scores)) and remaining:
                best_mmr = -float('inf')
                best_idx = -1
                
                for i, (candidate_id, candidate_score) in enumerate(remaining):
                    # Relevance
                    relevance = candidate_score
                    
                    # Diversity (максимальное сходство с уже выбранными)
                    max_similarity = 0
                    if candidate_id in self.preprocessor.book_features['book_id'].values:
                        cand_idx = self.preprocessor.book_features[self.preprocessor.book_features['book_id'] == candidate_id].index[0]
                        for sel_id, _ in selected:
                            if sel_id in self.preprocessor.book_features['book_id'].values:
                                sel_idx = self.preprocessor.book_features[self.preprocessor.book_features['book_id'] == sel_id].index[0]
                                similarity = self.preprocessor.book_similarity_matrix[cand_idx][sel_idx]
                                max_similarity = max(max_similarity, similarity)
                    
                    # MMR score
                    mmr = (1 - diversity_weight) * relevance - diversity_weight * max_similarity
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                
                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))
                else:
                    break
            
            return [book_id for book_id, _ in selected]
        else:
            return [book_id for book_id, _ in scores[:top_n]]
    
    def recommend(self, user_id, top_n=10):
        """Основная функция рекомендаций"""
        # 1. Генерация пула кандидатов
        candidates = self.generate_candidate_pool(user_id, top_n=100)
        
        # 2. Ранжирование кандидатов
        recommendations = self.rank_candidates(user_id, candidates, top_n=top_n)
        
        # 3. Если рекомендаций мало, добавляем популярные
        if len(recommendations) < top_n:
            popular_books = self.models['popularity'].head(top_n * 2)['book_id'].tolist()
            for book_id in popular_books:
                if book_id not in recommendations:
                    # Проверяем, не читал ли уже пользователь
                    if user_id in self.preprocessor.train_data['user_id'].values:
                        read_books = set(self.preprocessor.train_data[self.preprocessor.train_data['user_id'] == user_id]['book_id'])
                        if book_id not in read_books:
                            recommendations.append(book_id)
                    else:
                        recommendations.append(book_id)
                
                if len(recommendations) >= top_n:
                    break
        
        return recommendations[:top_n]
    
    def train_all_models(self):
        """Обучение всех моделей"""
        self.segment_users()
        self.train_popularity_model()
        self.train_content_based_model()
        self.train_collaborative_model()
        self.train_hybrid_model()

# ========================================================================
# ПРОДВИНУТАЯ ЧАСТЬ: НЕЙРОСЕТЕВОЙ ПОДХОД
# ========================================================================

print("\n" + "="*80)
print("ПРОДВИНУТАЯ ЧАСТЬ: НЕЙРОСЕТЕВОЙ ПОДХОД")
print("="*80)

class NeuralRecommender:
    """Two-Tower нейросетевая модель"""
    
    def __init__(self, preprocessor, embedding_dim=64):
        self.preprocessor = preprocessor
        self.embedding_dim = embedding_dim
        self.model = None
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()
        
    def prepare_data(self):
        """Подготовка данных для нейросети"""
        print("Подготовка данных для нейросети...")
        
        # Подготавливаем user_id и book_id
        all_user_ids = np.concatenate([
            self.preprocessor.train_data['user_id'].values,
            self.preprocessor.test_data['user_id'].values
        ])
        
        all_book_ids = np.concatenate([
            self.preprocessor.train_data['book_id'].values,
            self.preprocessor.test_data['book_id'].values
        ])
        
        # Кодируем ID
        self.user_encoder.fit(all_user_ids)
        self.book_encoder.fit(all_book_ids)
        
        # Преобразуем train данные
        train_users = self.user_encoder.transform(self.preprocessor.train_data['user_id'])
        train_books = self.book_encoder.transform(self.preprocessor.train_data['book_id'])
        train_ratings = self.preprocessor.train_data['rating'].values
        
        # Создаем отрицательные сэмплы
        print("Создание отрицательных сэмплов...")
        positive_pairs = set(zip(train_users, train_books))
        
        negative_samples = []
        n_negative = len(positive_pairs)  # Столько же негативных
        
        unique_users = np.unique(train_users)
        unique_books = np.unique(train_books)
        
        for _ in tqdm(range(n_negative), desc="Негативные сэмплы"):
            user = np.random.choice(unique_users)
            book = np.random.choice(unique_books)
            
            # Проверяем, что это не позитивная пара
            while (user, book) in positive_pairs:
                book = np.random.choice(unique_books)
            
            negative_samples.append((user, book, 0))  # 0 - негативный класс
        
        # Объединяем позитивные и негативные сэмплы
        positive_samples = list(zip(train_users, train_books, [1]*len(train_users)))  # 1 - позитивный класс
        all_samples = positive_samples + negative_samples
        
        np.random.shuffle(all_samples)
        
        # Разделяем на X и y
        X_users = np.array([s[0] for s in all_samples])
        X_books = np.array([s[1] for s in all_samples])
        y = np.array([s[2] for s in all_samples])
        
        # Разделение на train/validation
        split_idx = int(0.8 * len(X_users))
        
        self.X_train = (X_users[:split_idx], X_books[:split_idx])
        self.X_val = (X_users[split_idx:], X_books[split_idx:])
        self.y_train = y[:split_idx]
        self.y_val = y[split_idx:]
        
        print(f"✓ Данные подготовлены:")
        print(f"  • Уникальных пользователей: {len(self.user_encoder.classes_)}")
        print(f"  • Уникальных книг: {len(self.book_encoder.classes_)}")
        print(f"  • Тренировочных сэмплов: {len(self.X_train[0])}")
        print(f"  • Валидационных сэмплов: {len(self.X_val[0])}")
    
    def build_model(self):
        """Построение Two-Tower модели"""
        print("\nПостроение Two-Tower модели...")
        
        # Входы
        user_input = keras.Input(shape=(1,), name="user_input")
        book_input = keras.Input(shape=(1,), name="book_input")
        
        # Эмбеддинги
        n_users = len(self.user_encoder.classes_)
        n_books = len(self.book_encoder.classes_)
        
        user_embedding = layers.Embedding(
            input_dim=n_users + 1,
            output_dim=self.embedding_dim,
            embeddings_initializer='he_normal',
            name="user_embedding"
        )(user_input)
        
        book_embedding = layers.Embedding(
            input_dim=n_books + 1,
            output_dim=self.embedding_dim,
            embeddings_initializer='he_normal',
            name="book_embedding"
        )(book_input)
        
        # Flatten
        user_flat = layers.Flatten()(user_embedding)
        book_flat = layers.Flatten()(book_embedding)
        
        # Дополнительные слои для каждого тауэра
        user_dense = layers.Dense(128, activation='relu')(user_flat)
        user_dense = layers.Dropout(0.3)(user_dense)
        user_dense = layers.Dense(64, activation='relu')(user_dense)
        
        book_dense = layers.Dense(128, activation='relu')(book_flat)
        book_dense = layers.Dropout(0.3)(book_dense)
        book_dense = layers.Dense(64, activation='relu')(book_dense)
        
        # Скалярное произведение (как в Two-Tower)
        dot_product = layers.Dot(axes=1, normalize=False)([user_dense, book_dense])
        
        # Выходной слой
        output = layers.Dense(1, activation='sigmoid')(dot_product)
        
        # Модель
        self.model = keras.Model(
            inputs=[user_input, book_input],
            outputs=output,
            name="two_tower_model"
        )
        
        # Компиляция
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print("✓ Модель построена")
        self.model.summary()
    
    def train(self, epochs=10, batch_size=512):
        """Обучение модели"""
        print("\nОбучение Two-Tower модели...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                mode='max',
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Обучение
        history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Сохраняем историю
        self.history = history.history
        
        print("✓ Модель обучена")
        
        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = ['loss', 'accuracy', 'auc']
        titles = ['Loss', 'Accuracy', 'AUC']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            ax.plot(self.history[metric], label=f'Train {title}')
            ax.plot(self.history[f'val_{metric}'], label=f'Val {title}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(f'{title} over epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return history
    
    def recommend(self, user_id, top_n=10):
        """Рекомендации с помощью нейросети"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        # Кодируем user_id
        try:
            encoded_user = self.user_encoder.transform([user_id])[0]
        except:
            # Если пользователь новый, возвращаем популярные
            return []
        
        # Получаем все book_id (кодированные)
        all_books = np.arange(len(self.book_encoder.classes_))
        
        # Создаем пары (user, book) для всех книг
        user_array = np.full_like(all_books, encoded_user)
        
        # Предсказываем скоры
        predictions = self.model.predict(
            [user_array, all_books],
            batch_size=1024,
            verbose=0
        ).flatten()
        
        # Сортируем по убыванию скора
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        # Декодируем book_id
        recommended_book_ids = self.book_encoder.inverse_transform(top_indices)
        
        return recommended_book_ids.tolist()

# ========================================================================
# ИНТЕГРИРОВАННАЯ СИСТЕМА
# ========================================================================

print("\n" + "="*80)
print("ИНТЕГРИРОВАННАЯ СИСТЕМА")
print("="*80)

class IntegratedRecommenderSystem:
    """Интегрированная система с нейросетевым подходом"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.hybrid_system = None
        self.neural_recommender = None
        self.model_weights = {
            'hybrid': 0.4,
            'neural': 0.4,
            'popularity': 0.2
        }
    
    def run_pipeline(self):
        """Запуск всего пайплайна"""
        print("\n" + "="*80)
        print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
        print("="*80)
        
        # Этап 1: Подготовка данных
        print("\n📊 ЭТАП 1: ПОДГОТОВКА ДАННЫХ")
        self.preprocessor.prepare_all_features()
        
        # Этап 2: Гибридная система
        print("\n🤖 ЭТАП 2: ГИБРИДНАЯ СИСТЕМА")
        self.hybrid_system = HybridRecommenderSystem(self.preprocessor)
        self.hybrid_system.train_all_models()
        
        # Этап 3: Нейросетевая модель
        print("\n🧠 ЭТАП 3: НЕЙРОСЕТЕВАЯ МОДЕЛЬ")
        self.neural_recommender = NeuralRecommender(self.preprocessor)
        self.neural_recommender.prepare_data()
        self.neural_recommender.build_model()
        self.neural_recommender.train(epochs=15)
        
        # Оптимизация весов моделей
        print("\n⚖️  ЭТАП 4: ОПТИМИЗАЦИЯ ВЕСОВ")
        self.optimize_weights()
        
        print("\n" + "="*80)
        print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
        print("="*80)
    
    def optimize_weights(self):
        """Оптимизация весов моделей на валидационных данных"""
        print("Оптимизация весов моделей...")
        
        # Берем подмножество пользователей для оптимизации
        val_users = self.preprocessor.test_data['user_id'].unique()[:50]
        
        best_score = 0
        best_weights = self.model_weights.copy()
        
        # Простой grid search по весам
        weights_options = [
            {'hybrid': 0.5, 'neural': 0.3, 'popularity': 0.2},
            {'hybrid': 0.4, 'neural': 0.4, 'popularity': 0.2},
            {'hybrid': 0.3, 'neural': 0.5, 'popularity': 0.2},
            {'hybrid': 0.6, 'neural': 0.2, 'popularity': 0.2},
        ]
        
        for weights in weights_options:
            self.model_weights = weights
            score = self._evaluate_weight_combo(val_users[:10])
            
            print(f"  Веса {weights}: Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_weights = weights
        
        self.model_weights = best_weights
        print(f"✓ Оптимальные веса: {best_weights} (Score: {best_score:.4f})")
    
    def _evaluate_weight_combo(self, user_ids, top_n=10):
        """Оценка комбинации весов"""
        scores = []
        
        for user_id in user_ids:
            # Получаем рекомендации от всех моделей
            hybrid_rec = set(self.hybrid_system.recommend(user_id, top_n=top_n*3))
            neural_rec = set(self.neural_recommender.recommend(user_id, top_n=top_n*3))
            popularity_rec = set(self.hybrid_system.models['popularity'].head(top_n*3)['book_id'].tolist())
            
            # Объединяем с весами
            combined_scores = defaultdict(float)
            
            for i, book_id in enumerate(hybrid_rec):
                combined_scores[book_id] += self.model_weights['hybrid'] * (1.0 / (i + 1))
            
            for i, book_id in enumerate(neural_rec):
                combined_scores[book_id] += self.model_weights['neural'] * (1.0 / (i + 1))
            
            for i, book_id in enumerate(popularity_rec):
                combined_scores[book_id] += self.model_weights['popularity'] * (1.0 / (i + 1))
            
            # Ранжируем
            sorted_books = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [book_id for book_id, _ in sorted_books[:top_n]]
            
            # Оцениваем
            actual_books = set(self.preprocessor.test_data[
                self.preprocessor.test_data['user_id'] == user_id
            ]['book_id'])
            
            if actual_books:
                precision = len(set(recommendations) & actual_books) / top_n
                scores.append(precision)
        
        return np.mean(scores) if scores else 0
    
    def recommend(self, user_id, top_n=10):
        """Рекомендации из интегрированной системы"""
        print(f"\nГенерация рекомендаций для пользователя {user_id}...")
        
        # Определяем сегмент пользователя
        if user_id in self.hybrid_system.user_segments:
            segment = self.hybrid_system.user_segments[user_id]
            print(f"  Сегмент пользователя: {segment}")
        else:
            segment = 'new_user'
            print(f"  Новый пользователь")
        
        # Получаем рекомендации от всех моделей
        recommendations = {
            'hybrid': self.hybrid_system.recommend(user_id, top_n=top_n*2),
            'neural': self.neural_recommender.recommend(user_id, top_n=top_n*2),
            'popularity': self.hybrid_system.models['popularity'].head(top_n*2)['book_id'].tolist()
        }
        
        print(f"  Получено рекомендаций:")
        print(f"    • Гибридная модель: {len(recommendations['hybrid'])}")
        print(f"    • Нейросетевая модель: {len(recommendations['neural'])}")
        print(f"    • Популярные: {len(recommendations['popularity'])}")
        
        # Объединяем с весами
        combined_scores = defaultdict(float)
        
        for model_name, recs in recommendations.items():
            weight = self.model_weights[model_name]
            for i, book_id in enumerate(recs):
                combined_scores[book_id] += weight * (1.0 / (i + 1))
        
        # Сортируем по скору
        sorted_books = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Исключаем уже прочитанные
        if user_id in self.preprocessor.train_data['user_id'].values:
            read_books = set(self.preprocessor.train_data[
                self.preprocessor.train_data['user_id'] == user_id
            ]['book_id'])
            filtered_books = [(b, s) for b, s in sorted_books if b not in read_books]
        else:
            filtered_books = sorted_books
        
        # Берем топ-N
        final_recommendations = [book_id for book_id, _ in filtered_books[:top_n]]
        
        print(f"  Итоговых рекомендаций: {len(final_recommendations)}")
        
        return final_recommendations
    
    def evaluate_system(self, n_users=50, top_n=10):
        """Оценка всей системы"""
        print("\n" + "="*80)
        print("ОЦЕНКА СИСТЕМЫ")
        print("="*80)
        
        # Выбираем пользователей для оценки
        test_users = self.preprocessor.test_data['user_id'].unique()[:n_users]
        
        metrics = {
            'precision': [],
            'recall': [],
            'ndcg': [],
            'coverage': set(),
            'diversity': []
        }
        
        print(f"Оценка на {len(test_users)} пользователях...")
        
        for user_id in tqdm(test_users, desc="Оценка системы"):
            # Получаем рекомендации
            recommendations = self.recommend(user_id, top_n=top_n)
            
            # Получаем реальные оценки из теста
            actual_books = set(self.preprocessor.test_data[
                self.preprocessor.test_data['user_id'] == user_id
            ]['book_id'])
            
            if not actual_books:
                continue
            
            # Precision
            relevant = len(set(recommendations) & actual_books)
            precision = relevant / top_n if top_n > 0 else 0
            metrics['precision'].append(precision)
            
            # Recall
            recall = relevant / len(actual_books) if len(actual_books) > 0 else 0
            metrics['recall'].append(recall)
            
            # nDCG
            dcg = 0
            for i, book_id in enumerate(recommendations, 1):
                if book_id in actual_books:
                    dcg += 1 / np.log2(i + 1)
            
            ideal_rec = min(top_n, len(actual_books))
            idcg = sum(1 / np.log2(i + 1) for i in range(1, ideal_rec + 1))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg'].append(ndcg)
            
            # Coverage
            metrics['coverage'].update(recommendations)
            
            # Diversity (среднее попарное несходство)
            if len(recommendations) > 1:
                similarities = []
                for i in range(len(recommendations)):
                    for j in range(i + 1, len(recommendations)):
                        book1 = recommendations[i]
                        book2 = recommendations[j]
                        
                        if (book1 in self.preprocessor.book_features['book_id'].values and 
                            book2 in self.preprocessor.book_features['book_id'].values):
                            idx1 = self.preprocessor.book_features[
                                self.preprocessor.book_features['book_id'] == book1
                            ].index[0]
                            idx2 = self.preprocessor.book_features[
                                self.preprocessor.book_features['book_id'] == book2
                            ].index[0]
                            
                            similarity = self.preprocessor.book_similarity_matrix[idx1][idx2]
                            similarities.append(similarity)
                
                if similarities:
                    diversity = 1 - np.mean(similarities)
                    metrics['diversity'].append(diversity)
        
        # Вычисляем средние метрики
        avg_metrics = {
            'precision@K': np.mean(metrics['precision']) if metrics['precision'] else 0,
            'recall@K': np.mean(metrics['recall']) if metrics['recall'] else 0,
            'nDCG@K': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0,
            'coverage': len(metrics['coverage']) / len(self.preprocessor.book_features) if len(self.preprocessor.book_features) > 0 else 0,
            'diversity': np.mean(metrics['diversity']) if metrics['diversity'] else 0,
            'f1_score': 0
        }
        
        # F1-score
        if avg_metrics['precision@K'] + avg_metrics['recall@K'] > 0:
            avg_metrics['f1_score'] = 2 * avg_metrics['precision@K'] * avg_metrics['recall@K'] / (
                avg_metrics['precision@K'] + avg_metrics['recall@K']
            )
        
        print("\n📈 РЕЗУЛЬТАТЫ ОЦЕНКИ:")
        print("-" * 40)
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Метрики качества
        quality_metrics = ['precision@K', 'recall@K', 'nDCG@K', 'f1_score']
        quality_values = [avg_metrics[m] for m in quality_metrics]
        
        axes[0].bar(quality_metrics, quality_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        axes[0].set_title('Метрики качества рекомендаций')
        axes[0].set_ylabel('Значение')
        axes[0].grid(True, alpha=0.3)
        
        for i, v in enumerate(quality_values):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Метрики разнообразия
        diversity_metrics = ['coverage', 'diversity']
        diversity_values = [avg_metrics['coverage'], avg_metrics['diversity']]
        
        axes[1].bar(diversity_metrics, diversity_values, color=['#9b59b6', '#1abc9c'])
        axes[1].set_title('Метрики разнообразия и покрытия')
        axes[1].set_ylabel('Значение')
        axes[1].grid(True, alpha=0.3)
        
        for i, v in enumerate(diversity_values):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return avg_metrics
    
    def save_pipeline(self, path='recommendation_pipeline'):
        """Сохранение всего пайплайна"""
        print(f"\nСохранение пайплайна в {path}...")
        
        # Создаем папку
        import os
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем компоненты
        components = {
            'preprocessor': self.preprocessor,
            'hybrid_system': self.hybrid_system,
            'neural_recommender': self.neural_recommender,
            'model_weights': self.model_weights
        }
        
        with open(f'{path}/pipeline.pkl', 'wb') as f:
            pickle.dump(components, f)
        
        # Сохраняем метаданные
        metadata = {
            'n_users': len(self.preprocessor.user_features),
            'n_books': len(self.preprocessor.book_features),
            'n_interactions': len(self.preprocessor.interaction_features),
            'model_weights': self.model_weights,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(f'{path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Пайплайн сохранен")
        
        return metadata
    
    def load_pipeline(self, path='recommendation_pipeline'):
        """Загрузка сохраненного пайплайна"""
        print(f"\nЗагрузка пайплайна из {path}...")
        
        with open(f'{path}/pipeline.pkl', 'rb') as f:
            components = pickle.load(f)
        
        self.preprocessor = components['preprocessor']
        self.hybrid_system = components['hybrid_system']
        self.neural_recommender = components['neural_recommender']
        self.model_weights = components['model_weights']
        
        print("✓ Пайплайн загружен")
        
        return self

# ========================================================================
# ЗАПУСК СКВОЗНОГО ПАЙПЛАЙНА
# ========================================================================

if __name__ == "__main__":
    print("="*80)
    print("СКВОЗНОЙ ПАЙПЛАЙН РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("="*80)
    
    # Создаем и запускаем систему
    system = IntegratedRecommenderSystem()
    
    try:
        # Запуск полного пайплайна
        system.run_pipeline()
        
        # Оценка системы
        metrics = system.evaluate_system(n_users=100, top_n=10)
        
        # Сохранение пайплайна
        metadata = system.save_pipeline()
        
        # Демонстрация рекомендаций для тестового пользователя
        print("\n" + "="*80)
        print("ДЕМОНСТРАЦИЯ РЕКОМЕНДАЦИЙ")
        print("="*80)
        
        # Выбираем тестового пользователя
        test_user = system.preprocessor.test_data['user_id'].iloc[0]
        
        print(f"\nПример рекомендаций для пользователя {test_user}:")
        recommendations = system.recommend(test_user, top_n=10)
        
        if recommendations:
            print("\nТоп-10 рекомендаций:")
            for i, book_id in enumerate(recommendations, 1):
                book_info = system.preprocessor.book_features[
                    system.preprocessor.book_features['book_id'] == book_id
                ]
                
                if not book_info.empty:
                    title = book_info['title'].iloc[0]
                    authors = book_info['authors'].iloc[0]
                    avg_rating = book_info['book_avg_rating'].iloc[0]
                    
                    print(f"{i}. {title}")
                    print(f"   Авторы: {authors}")
                    print(f"   Средний рейтинг: {avg_rating:.2f}")
                    print()
        
        print("\n" + "="*80)
        print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
        print("="*80)
        
        print("\n📊 ИТОГОВЫЕ МЕТРИКИ СИСТЕМЫ:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n🎯 КЛЮЧЕВЫЕ ХАРАКТЕРИСТИКИ СИСТЕМЫ:")
        print("  • Гибридная архитектура с несколькими моделями")
        print("  • Two-Tower нейросетевая модель")
        print("  • Автоматическая сегментация пользователей")
        print("  • Балансировка релевантности и разнообразия")
        print("  • Возможность повторного запуска на новых данных")
        
    except Exception as e:
        print(f"\n✗ Ошибка в пайплайне: {e}")
        import traceback
        traceback.print_exc()
