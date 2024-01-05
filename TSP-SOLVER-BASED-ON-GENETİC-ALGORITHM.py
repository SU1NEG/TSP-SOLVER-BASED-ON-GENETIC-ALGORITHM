import random  # Rastgele değerler oluşturmak için random modülünü içe aktar
import math    # Matematiksel fonksiyonlar için math modülünü içe aktar
import copy    # Nesnelerin kopyalarını oluşturmak için copy modülünü içe aktar
import matplotlib.pyplot as plt  # Grafik çizimleri için matplotlib kütüphanesini içe aktar

class Point:
    def __init__(self, id, x, y):
        # Point sınıfının kurucusu, bir noktayı id, x-koordinatı ve y-koordinatı ile başlatır
        self.id = id
        self.x = x
        self.y = y

    def __str__(self):
        # Point nesnesinin dize temsilini döndüren metot, yazdırmak için kullanılır
        return f"Nokta{{id={self.id}, x={self.x}, y={self.y}}}"

    def __eq__(self, other):
        # Point nesneleri için eşitlik karşılaştırma metodu
        # Diğer nesne bir Point ise ve aynı id'ye sahipse True döner
        return isinstance(other, Point) and self.id == other.id  

def calculate_distance(p1, p2):
    # İki nokta arasındaki uzaklığı hesaplayan fonksiyon
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)  # Uzaklık formülü: sqrt((dx^2) + (dy^2))

def calculate_fitness(chromosome):
    # Verilen kromozomun fitness (uygunluk) değerini hesaplayan fonksiyon
    total_distance = 0.0  # Toplam uzaklık başlangıçta sıfır olarak ayarlanır
    num_points = len(chromosome)  # Kromozomdaki nokta sayısı

    for i in range(num_points):
        current_point = chromosome[i]
        next_point = chromosome[(i + 1) % num_points]

        # Her iki ardışık nokta arasındaki uzaklığı hesapla ve toplam uzaklığa ekle
        distance = calculate_distance(current_point, next_point)
        total_distance += distance

    return total_distance  # Kromozomun toplam uzaklık temelinde uygunluk değeri

def rank_based_selection(chromosomes, fitness_values, num_parents):
    selected_chromosomes = []  # Seçilen kromozomlar

    # Fitness_values'ın tek bir değer olup olmadığını kontrol ediyoruz (int veya float)
    if isinstance(fitness_values, (int, float)):
        fitness_values = [fitness_values] * len(chromosomes)

    # Kromozomları uygunluk değerlerine göre sıralıyoruz. (rank)
    ranked_population = sorted(zip(chromosomes, fitness_values), key=lambda x: x[1], reverse=True)
    ranked_chromosomes, _ = zip(*ranked_population)

    # Sıralamaya göre her bir kromozom için olasılığı hesaplıyoruz.
    total_chromosomes = len(chromosomes)  # Toplam kromozom sayısı
    cumulative_probability = 0.0  # Kumulatif olasılık
    selection_probabilities = []  # Seçim olasılıklarını saklamak için liste oluşturma
    for i in range(total_chromosomes):  # Her bir kromozom için döngü
        probability = (i + 1) / total_chromosomes  # Sıralamaya dayalı olarak olasılık hesaplama
        selection_probabilities.append(probability)  # Her bir olasılığı listeye ekleme
        cumulative_probability += probability  # Kumulatif olasılığı güncelleme

    # Rulet çarkı oluşturmak için olasılıkları normalleştiriyoruz.
    selection_probabilities = [p / cumulative_probability for p in selection_probabilities]

    # Hesaplanan olasılıklara dayalı bir rulet çarkı oluşturuyoruz.
    thresholds = random.choices(selection_probabilities, k=num_parents)
    for threshold in thresholds:
        cumulative_probability_sum = 0.0  # Kumulatif olasılık toplamı
        for i, probability in enumerate(selection_probabilities):  # Her bir kromozom için döngü
            cumulative_probability_sum += probability  # Kumulatif olasılık toplamını güncelleme
            if cumulative_probability_sum >= threshold:  # Eşik değerine ulaşıldığında
                selected_chromosomes.append(ranked_chromosomes[i])  # Seçilen kromozomu belirleme
                break

    return selected_chromosomes  # Seçilen kromozomları geri döndürme

def roulette_based_selection(chromosomes, num_parents):
    selected_chromosomes = []  # Seçilen kromozomlar

    # Fitness_values'ın tek bir değer olup olmadığını kontrol ediyoruz (int veya float)
    fitness_values = [calculate_fitness(chromosome) for chromosome in chromosomes]

    # Kromozomları uygunluk değerlerine göre sıralıyoruz.
    ranked_population = sorted(zip(chromosomes, fitness_values), key=lambda x: x[1], reverse=True)
    ranked_chromosomes, _ = zip(*ranked_population)

    # Sıralamaya göre her bir kromozom için olasılığı hesaplıyoruz.
    total_chromosomes = len(chromosomes)  # Toplam kromozom sayısı
    cumulative_probability = 0.0  # Kumulatif olasılık
    selection_probabilities = []  # Seçim olasılıklarını saklamak için liste oluşturma
    for i in range(total_chromosomes):  # Her bir kromozom için döngü
        probability = (i + 1) / total_chromosomes  # Sıralamaya dayalı olarak olasılık hesaplama
        selection_probabilities.append(probability)  # Her bir olasılığı listeye ekleme
        cumulative_probability += probability  # Kumulatif olasılığı güncelleme

    # Rulet çarkı oluşturmak için olasılıkları normalleştiriyoruz.
    selection_probabilities = [p / cumulative_probability for p in selection_probabilities]

    # Hesaplanan olasılıklara dayalı bir rulet çarkı oluşturuyoruz.
    thresholds = random.choices(selection_probabilities, k=num_parents)
    for threshold in thresholds:
        cumulative_probability_sum = 0.0  # Kumulatif olasılık toplamı
        for i, probability in enumerate(selection_probabilities):  # Her bir kromozom için döngü
            cumulative_probability_sum += probability  # Kumulatif olasılık toplamını güncelleme
            if cumulative_probability_sum >= threshold:  # Eşik değerine ulaşıldığında
                selected_chromosomes.append(ranked_chromosomes[i])  # Seçilen kromozomu belirleme
                break

    return selected_chromosomes  # Seçilen kromozomları geri döndürme

def cycle_crossover(parent1, parent2):
    size = len(parent1)  # Kromozom boyutu
    child = [None] * size  # Boyutu parent1'in boyutu kadar olan None değerleri içeren bir liste oluşturma
    visited_indices = set()  # Ziyaret edilen indeksleri tutmak için bir küme oluşturma

    is_parent1 = True  # Hangi ebeveynin kullanılacağını belirlemek için bir durum
    index = 0  # Başlangıç indeksi

    while len(visited_indices) < size:  # Tüm indeksler ziyaret edilene kadar döngüyü devam ettir
        visited_indices.add(index)  # Şu anki indeksi ziyaret edildi olarak işaretle
        if is_parent1:
            child[index] = parent1[index]  # Çocuğa parent1'den geni ekle
            next_index = parent2.index(parent1[index])  # Parent2'de bu genin indeksini bul
            index = next_index  # Bir sonraki indeksi ayarla
        else:
            child[index] = parent2[index]  # Çocuğa parent2'den geni ekle
            next_index = parent1.index(parent2[index])  # Parent1'de bu genin indeksini bul
            index = next_index  # Bir sonraki indeksi ayarla

        if index in visited_indices:  # Eğer indeks ziyaret edilmişse
            next_unvisited_index = find_unvisited_index(visited_indices, size)  # Ziyaret edilmemiş bir sonraki indeksi bul
            if next_unvisited_index != -1:
                index = next_unvisited_index  # Bir sonraki indeksi güncelle
                is_parent1 = not is_parent1  # Ebeveynleri değiştir

    return child  # Oluşturulan çocuk kromozomunu geri döndür

def find_unvisited_index(visited_indices, size):
    for i in range(size):
        if i not in visited_indices:
            return i
    return -1

def insert_mutation(chromosome):
    # Rastgelelik için random sınıfının kullanımı
    random_instance = random.Random()

    # Kromozom uzunluğuna göre rastgele bir indeks seçme
    index1 = random_instance.randint(0, len(chromosome) - 1)
    index2 = random_instance.randint(0, len(chromosome) - 1)

    # İki farklı indeks seçmek için tekrar seçim yapılır
    while index1 == index2:
        # Eğer indeksler aynıysa, ikinci indeksi tekrar seçme
        index2 = random_instance.randint(0, len(chromosome) - 1)

    # İki indeksin minimum ve maksimum değerini bulma
    min_index, max_index = min(index1, index2), max(index1, index2)

    # Kromozomdan ikinci indeksteki geni çıkarma
    gene_to_move = chromosome.pop(max_index)

    # Çıkarılan geni, birinci indeksten sonraki yere ekleme
    chromosome.insert(min_index + 1, gene_to_move)

    return chromosome

def random_slide_mutation(chromosome):
    # Rastgelelik için random sınıfının kullanımı
    random_instance = random.Random()

    # Rastgele bir başlangıç ve bitiş indeksi seçme
    start_index = random_instance.randint(0, len(chromosome) - 1)  # Kromozom uzunluğuna göre rastgele bir başlangıç indeksi seçme
    end_index = random_instance.randint(0, len(chromosome) - 1)  # Kromozom uzunluğuna göre rastgele bir bitiş indeksi seçme

    # Başlangıç ve bitiş indekslerini sıralama
    min_index, max_index = min(start_index, end_index), max(start_index, end_index)  # İki indeksin minimum ve maksimum değerini bulma

    # Genlerin alt kümesini alıp bu alt küme genlerini rastgele bir konuma kaydırma
    sub_list = chromosome[min_index:max_index + 1]  # Belirli bir aralıktaki genlerin alt listesini alma
    chromosome = [gene for gene in chromosome if gene not in sub_list]  # Kromozomdaki alt listeyi kaldırma

    insert_index = random_instance.randint(0, len(chromosome))  # Kromozomda rastgele bir konum belirleme
    chromosome[insert_index:insert_index] = sub_list  # Kromozoma, rastgele belirlenen konuma alt listeyi ekleme

    return chromosome


def create_new_generation(previous_generation, best_chromosome, population_size=100):
    new_generation = [copy.deepcopy(best_chromosome)]

    num_parents_rank = population_size // 2
    num_parents_roulette = population_size - num_parents_rank

    # Sıralamaya dayalı seçim
    parents_rank = rank_based_selection(previous_generation, calculate_fitness(best_chromosome), num_parents_rank)
    new_generation.extend(parents_rank)

    # Rulet çarkı seçimi
    parents_roulette = roulette_based_selection(previous_generation, num_parents_roulette)
    new_generation.extend(parents_roulette)

    for _ in range(population_size - 1):  # Zaten en iyi kromozom eklenmiş durumda
        parent1, parent2 = random.sample(new_generation, 2)

        # çaprazlama
        child = cycle_crossover(parent1, parent2)
        
        # Mutasyon
        if random.random() < 0.5:
            child = insert_mutation(child)
        else:
            child = random_slide_mutation(child)
        
        
        new_generation.append(child)

    return new_generation

def plot_points(chromosome):
    x_values = [point.x for point in chromosome]
    y_values = [point.y for point in chromosome]

    # Kromozomdaki noktaları birleştiren hatları çiz
    plt.figure()
    plt.plot(x_values + [x_values[0]], y_values + [y_values[0]], c='blue', marker='o', linestyle='-')
    plt.title('Best Chromosome Points Connected')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def plot_fitness_progress(generations):
    best_fitness_values = [min(map(calculate_fitness, generation)) for generation in generations]
    generation_numbers = list(range(1, len(generations) + 1))

    plt.plot(generation_numbers, best_fitness_values, marker='o')
    plt.title('Genetic Algorithm Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.show()

    # En iyi kromozomu bul ve onun noktalarını çiz
    best_chromosome = min(generations[-1], key=calculate_fitness)
    plot_points(best_chromosome)



def main():
    points = []  # Noktaları depolamak için bir liste oluşturuluyor

    # Dosyadan okuma
    is_data_section = False  # Veri bölümüne geçildiğini belirlemek için bir bayrak
    with open("C:\\Users\\Muhammed\\Downloads\\att48.txt", "r") as file:
        for line in file:
            if line.startswith("NODE_COORD_SECTION"):
                is_data_section = True  # Veri bölümüne geçildi
                continue


            if is_data_section:
                 # Veri bölümünde ise
                parts = line.split() # Satırdaki parçaları ayır
                if len(parts) == 3:
                      # Eğer parça sayısı üç ise (isim, x, y)
                    name = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    points.append(Point(name, x, y)) # Noktalar listesine yeni bir nokta ekle

    # Noktaları başlangıçta bir kere karıştır
    random.shuffle(points)

    # Nesiller listesini sıfırla
    generations = []

    # İlk nesil
    chromosomes = [list(points) for _ in range(100)] # 100 adet kromozom oluştur
    generations.append(chromosomes)

    # 100 nesil boyunca yinele
    for generation_num in range(100):
        current_generation = generations[-1]
        best_chromosome = min(current_generation, key=calculate_fitness)
        print(f"Generation {generation_num + 1}, Best Fitness: {calculate_fitness(best_chromosome)}")

        # Yeni bir nesil yarat
        new_generation = create_new_generation(current_generation, best_chromosome)

        generations.append(new_generation)

    plot_fitness_progress(generations)

if __name__ == "__main__":
    main()
