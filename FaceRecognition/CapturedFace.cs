using Emgu.CV;
using Emgu.CV.Structure;
using System.Windows.Media.Imaging;

namespace FaceRecognition
{
    public class FaceData
    {
        public string Label { get; set; }
        public BitmapImage ActualImage { get; set; }
        public Image<Gray, byte> CVImage { get; set; }
    }
}
