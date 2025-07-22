# Implementasi Pose Estimation untuk Pemodelan Gerak Tari Tradisional Indonesia pada Robot Humanoid ROBOTIS-OP3

## Deskripsi

Repositori ini merupakan bagian dari penelitian skripsi yang mengembangkan sistem estimasi pose 3D dan imitasi gerakan tari tradisional Indonesia pada robot humanoid ROBOTIS-OP3.

Sistem ini terdiri dari beberapa komponen utama:

- **Pose Estimation 3D** menggunakan kombinasi model Metrabs dan DeciWatch.
- **Filtering Temporal** menggunakan DeciWatch untuk meningkatkan akurasi dan kelancaran estimasi pose.
- **Inverse Kinematics Transform** untuk konversi pose manusia menjadi sudut servo robot.
- **Kontrol Robot Humanoid** berbasis komunikasi HTTP POST untuk menggerakkan ROBOTIS-OP3 secara otomatis.
- **Aplikasi Website** sebagai antarmuka terpadu dari input video hingga eksekusi gerakan pada robot.

## Fitur

- Estimasi pose 3D dari video input.
- Konversi pose ke sudut servo dengan pembatasan sudut untuk menghindari self-collision.
- Visualisasi perbandingan pose raw dan smooth.
- Eksekusi gerakan tari pada robot humanoid ROBOTIS-OP3.
- Antarmuka berbasis website untuk kemudahan operasional.

## Struktur Direktori

```
.
├── visualization/            # Visualisasi pose, skeleton, dan sudut servo
├── website/                  # Aplikasi web untuk pipeline end-to-end
├── data_video/               # Video input dan hasil output robot
├── models/                   # Model Metrabs & DeciWatch (pretrained & custom)
├── scripts/                  # Script konversi, evaluasi, dan training
└── README.md
```

## Batasan Sistem

- Robot diam di tempat, tanpa pergerakan translasi kaki.
- Pergerakan terbatas pada badan bagian atas (lengan dan kepala).
- Dataset training menggunakan dataset publik (AIST++).
- Tidak terdapat sinkronisasi langsung dengan musik.

## Limitasi Sudut Servo

| Joint          | Range (derajat) |
| -------------- | --------------- |
| LShoulderPitch | -180 \~ 180     |
| LShoulderRoll  | -100 \~ 100     |
| LElbowTwist    | -180 \~ 180     |
| LElbowFlexion  | -180 \~ 110     |
| RShoulderPitch | -180 \~ 180     |
| RShoulderRoll  | -100 \~ 100     |
| RElbowTwist    | -180 \~ 180     |
| RElbowFlexion  | -180 \~ 110     |

## Dataset

- **AIST++ Motion Dataset** untuk training DeciWatch.
- Video tari tradisional Indonesia sebagai data pengujian.

## Penulis

luthfai\
Politeknik Negeri Malang\
2025

