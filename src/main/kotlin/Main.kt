import net.tzolov.cv.mtcnn.MtcnnService
import net.tzolov.cv.mtcnn.MtcnnUtil
import java.io.File
import javax.imageio.ImageIO

fun main() {
    val mtcnnService = MtcnnService()
    val inputImage = MtcnnUtil.to3ByteBGR(ImageIO.read(File("pivotal-ipo-nyse.jpg")))!!
    println("Loaded image as BGR: ${inputImage.width}x${inputImage.height} ${inputImage.type}")
    val faceAnnotations = mtcnnService.faceDetection(inputImage)
    val annotatedImage = MtcnnUtil.drawFaceAnnotations(inputImage, faceAnnotations)
    ImageIO.write(annotatedImage, "png", File("./AnnotatedImage.png"))
}
