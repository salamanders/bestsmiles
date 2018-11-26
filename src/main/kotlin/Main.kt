import mu.KotlinLogging
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.awt.Rectangle
import java.nio.file.Paths


private val logger = KotlinLogging.logger {}

private val imageNetMeanValuesRGB = Scalar(103.93, 116.77, 123.68) // TBD if this needs to be changed for each model

private val imageNetSize = 299 // Does this mean it scales down the input to this size? Or windows over the image?

fun findStuff(net: Net, filename: String): List<Pair<Rectangle, String>> {

    val frame = Imgcodecs.imread(filename)!!
    val frameWidth = frame.cols()
    val frameHeight = frame.rows()

    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB) // Strip out any alpha
    Imgproc.resize(frame, frame, Size(300.0, 300.0))

    // TODO: This looks like it needs plenty of knowledge.
    // swapRB = blobFromImage expects BRG, but the mean is a RGB scalar.  So if you are passing in the image AND the mean, you might need this.
    //val blob = Dnn.blobFromImage(frame, 1.0, Size(IN_WIDTH, IN_HEIGHT), Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/false, /*crop*/false)
    val blob = Dnn.blobFromImage(frame,
            1.0, Size(300.0, 300.0),
            Scalar(104.0, 177.0, 123.0), // https://github.com/prouast/heartbeat/blob/master/RPPG.cpp#L194
            true, false)!!
    logger.info { "Made blob." }

    net.setInput(blob)
    val detectionsUnshaped = net.forward()
    detectionsUnshaped.reshape(1, detectionsUnshaped.total().toInt() / 7)!!.let { detections ->
        logger.info { "Got detections: ${detections.rows()}" }
        return (0 until detections.rows()).mapNotNull { i ->
            if (detections.get(i, 2)[0] < 0.2) {
                null
            } else {
                val classId = detections.get(i, 1)[0].toInt() // TODO: How does the class work?
                val left = (detections.get(i, 3)[0] * frameWidth).toInt()
                val top = (detections.get(i, 4)[0] * frameHeight).toInt()
                val right = (detections.get(i, 5)[0] * frameWidth).toInt()
                val bottom = (detections.get(i, 6)[0] * frameHeight).toInt()
                // Draw rectangle around detected object.
                //Imgproc.rectangle(frame, Point(left.toDouble(), top.toDouble()), Point(right.toDouble(), bottom.toDouble()), Scalar(0.0, 255.0, 0.0))
                Pair(Rectangle(left, top, right - left, bottom - top), "face")
            }
        }
    }
}


fun main() {
    logger.info { "Starting app." }

    nu.pattern.OpenCV.loadShared() // Required first by org.openpnp
    System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME)

    val prototxt = Paths.get("src/main/resources/res10_300x300_ssd_iter_140000.caffemodel")!!
    val caffeModel = Paths.get("src/main/resources/deploy.prototxt")!!

    require(prototxt.toFile().canRead()) { "Missing ${prototxt.toAbsolutePath()}" }
    require(caffeModel.toFile().canRead()) { "Missing ${caffeModel.toAbsolutePath()}" }

    val net = Dnn.readNetFromCaffe(prototxt.toString(), caffeModel.toString())!!
    logger.info { "Network loaded successfully" }

    // Get a new frame
    val faces = findStuff(net, "300.png") // 300x300 cropped image
    println(faces.joinToString())
}